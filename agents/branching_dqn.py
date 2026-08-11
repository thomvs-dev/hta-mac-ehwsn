"""Variable-cardinality Branching Dueling C51 agent for HTA-MAC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .architectures import (
    EquivariantSetBranchingC51,
    GlobalBranchingDuelingC51,
    IndependentDuelingC51,
)
from .budget_projection import project_slot_budget
from .prioritized_replay import PrioritizedReplay


class BranchingDuelingC51(nn.Module):
    """Weight-tied per-node branches with dueling distributional heads."""

    def __init__(
        self,
        input_dim: int = 50,
        hidden_dim: int = 128,
        actions: int = 4,
        atoms: int = 51,
        v_min: float = -30.0,
        v_max: float = 30.0,
    ):
        super().__init__()
        self.actions = actions
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max
        self.register_buffer("support", torch.linspace(v_min, v_max, atoms))
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, atoms)
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, actions * atoms),
        )

    def forward(self, state, mask=None):
        hidden = self.trunk(state)
        value = self.value(hidden).unsqueeze(2)
        advantage = self.advantage(hidden).view(
            *hidden.shape[:-1], self.actions, self.atoms
        )
        logits = value + advantage - advantage.mean(dim=2, keepdim=True)
        return F.log_softmax(logits, dim=-1)

    def q_values(self, state, mask=None):
        log_probabilities = self.forward(state, mask)
        return (log_probabilities.exp() * self.support).sum(dim=-1)


@dataclass
class BranchingAgentConfig:
    input_dim: int = 50
    actions: int = 4
    budget: int = 24
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 32
    replay_capacity: int = 20000
    warmup: int = 256
    target_update_steps: int = 250
    v_min: float = -30.0
    v_max: float = 30.0
    atoms: int = 51
    max_branches: int = 100
    architecture: str = "shared_branching"
    state_schema: str = "phase2c_v1"
    embedding_start_dim: int = 18
    normalize_input_blocks: bool = False
    hybrid_harvest_max_j: float = 1.0
    trajectory_low_block: tuple[float, ...] = ()
    trajectory_high_block: tuple[float, ...] = ()
    trajectory_loss_weight: float = 0.0
    concavity_loss_weight: float = 0.0
    trajectory_margin_fraction: float = 0.05
    demonstration_margin_loss_weight: float = 0.0
    demonstration_margin: float = 0.8
    precision: str = "fp32"
    reward_scale: float = 1.0


class BranchingDQNAgent:
    def __init__(self, config: BranchingAgentConfig, device="cpu"):
        self.cfg = config
        self.device = torch.device(device)
        if config.precision not in {"fp32", "bf16"}:
            raise ValueError(f"unsupported precision: {config.precision}")
        if config.precision == "bf16" and self.device.type != "cuda":
            raise ValueError("bf16 training requires a CUDA device")
        kwargs = dict(
            input_dim=config.input_dim,
            actions=config.actions,
            atoms=config.atoms,
            v_min=config.v_min,
            v_max=config.v_max,
        )
        if config.architecture == "legacy_weight_tied":
            network_type = BranchingDuelingC51
        elif config.architecture == "shared_branching":
            network_type = GlobalBranchingDuelingC51
            kwargs["max_branches"] = config.max_branches
        elif config.architecture == "equivariant_set_branching":
            network_type = EquivariantSetBranchingC51
            kwargs["budget"] = config.budget
            kwargs["max_branches"] = config.max_branches
        elif config.architecture == "independent_dqns":
            network_type = IndependentDuelingC51
            kwargs["max_branches"] = config.max_branches
        else:
            raise ValueError(f"unsupported architecture: {config.architecture}")
        self.online = network_type(**kwargs).to(self.device)
        self.target = network_type(**kwargs).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=config.learning_rate
        )
        self.replay = PrioritizedReplay(config.replay_capacity)
        self.train_steps = 0
        self.last_loss_terms = None

    @staticmethod
    def _categorical_output_layers(network):
        """Return only the final value/advantage atom projection layers."""
        layers = []
        if isinstance(network, GlobalBranchingDuelingC51):
            layers.append(network.value[-1])
            layers.extend(head[-1] for head in network.advantages)
        elif isinstance(network, EquivariantSetBranchingC51):
            layers.extend((network.value[-1], network.advantage[-1]))
        elif isinstance(network, IndependentDuelingC51):
            for local in network.networks:
                layers.extend((local.value[-1], local.advantage[-1]))
        elif isinstance(network, BranchingDuelingC51):
            layers.extend((network.value[-1], network.advantage[-1]))
        else:
            raise TypeError(f"unsupported network type: {type(network)!r}")
        if not layers or not all(isinstance(layer, nn.Linear) for layer in layers):
            raise RuntimeError("categorical output-layer discovery failed")
        return layers

    def reinitialize_categorical_outputs(self, seed: int | None = None):
        """Reset atom projections while preserving every upstream parameter."""
        if seed is not None:
            devices = (
                [self.device.index or torch.cuda.current_device()]
                if self.device.type == "cuda"
                else []
            )
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(seed))
                for layer in self._categorical_output_layers(self.online):
                    layer.reset_parameters()
        else:
            for layer in self._categorical_output_layers(self.online):
                layer.reset_parameters()
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=self.cfg.learning_rate
        )
        self.replay = PrioritizedReplay(self.cfg.replay_capacity)
        self.train_steps = 0
        self.last_loss_terms = None
    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.cfg.precision == "bf16",
        )

    def _transform_state_tensor(self, state):
        """Scale physical forecast moments and the inherited embedding."""
        if not self.cfg.normalize_input_blocks:
            return state
        transformed = state.clone()
        harvest_scale = max(float(self.cfg.hybrid_harvest_max_j), 1e-12)
        transformed[..., 1] = transformed[..., 1] / harvest_scale
        transformed[..., 2] = transformed[..., 2] / (harvest_scale ** 2)
        embedding_start = int(self.cfg.embedding_start_dim)
        if not 18 <= embedding_start <= transformed.shape[-1]:
            raise ValueError("embedding start is incompatible with state schema")
        if transformed.shape[-1] > embedding_start:
            transformed[..., embedding_start:] = F.layer_norm(
                transformed[..., embedding_start:],
                (transformed.shape[-1] - embedding_start,),
            )
        return transformed

    def q_values_tensor(self, state, mask=None, *, target=False):
        network = self.target if target else self.online
        with self._autocast():
            return network.q_values(self._transform_state_tensor(state), mask)

    @staticmethod
    def _concavity_loss(q_values, masks):
        """Penalize increasing marginal Q gains for successive slots."""
        marginal = q_values[..., 1:] - q_values[..., :-1]
        violations = F.relu(marginal[..., 1:] - marginal[..., :-1])
        valid = masks.unsqueeze(-1).expand_as(violations)
        if not torch.any(valid):
            return q_values.sum() * 0.0
        return violations[valid].mean()

    def _trajectory_order_loss(self, raw_states, masks):
        """Require high-harvest counterfactuals to have no smaller slot gains."""
        low_block = self.cfg.trajectory_low_block
        high_block = self.cfg.trajectory_high_block
        if len(low_block) != 14 or len(high_block) != 14:
            return raw_states.sum() * 0.0
        valid_samples = torch.any(masks, dim=1)
        if not torch.any(valid_samples):
            return raw_states.sum() * 0.0
        selected = torch.argmax(masks.to(torch.int64), dim=1)
        low_states = raw_states[valid_samples].clone()
        high_states = raw_states[valid_samples].clone()
        selected = selected[valid_samples]
        rows = torch.arange(selected.shape[0], device=raw_states.device)
        low_values = raw_states.new_tensor(low_block)
        high_values = raw_states.new_tensor(high_block)
        low_states[rows, selected, 1:15] = low_values
        high_states[rows, selected, 1:15] = high_values
        counterfactual_masks = masks[valid_samples]
        low_q = self.q_values_tensor(low_states, counterfactual_masks)
        high_q = self.q_values_tensor(high_states, counterfactual_masks)
        low_q = low_q[rows, selected]
        high_q = high_q[rows, selected]
        low_marginal = low_q[:, 1:] - low_q[:, :-1]
        high_marginal = high_q[:, 1:] - high_q[:, :-1]
        scale = torch.cat((low_marginal, high_marginal), dim=1).detach()
        scale = scale.abs().median().clamp_min(1e-3)
        margin = float(self.cfg.trajectory_margin_fraction) * scale
        return F.relu(margin + low_marginal - high_marginal).mean()

    def _project(
        self,
        q_values,
        mask,
        *,
        fill_budget=False,
        caps=None,
        budget=None,
        tie_break_priorities=None,
    ):
        q = np.asarray(q_values, dtype=np.float64).copy()
        mask = np.asarray(mask, dtype=bool)
        if caps is None:
            caps = np.full(len(q), self.cfg.actions - 1, dtype=np.int64)
        caps = np.clip(
            np.asarray(caps, dtype=np.int64), 0, self.cfg.actions - 1
        )
        caps[~mask] = 0
        q[~mask, 1:] = -1e9
        levels = np.arange(self.cfg.actions)[None, :]
        q[levels > caps[:, None]] = -1e9
        allocation = project_slot_budget(
            q,
            self.cfg.budget if budget is None else int(budget),
            stop_at_nonpositive_gain=not fill_budget,
            tie_break_priorities=tie_break_priorities,
        )
        allocation = np.minimum(allocation, caps)
        allocation[~mask] = 0
        return allocation

    def act(
        self,
        state,
        mask,
        epsilon=0.0,
        caps=None,
        budget=None,
        tie_break_priorities=None,
    ):
        if epsilon > 0.0 and np.random.random() < epsilon:
            random_q = np.random.normal(
                size=(len(state), self.cfg.actions)
            )
            random_q[:, 1:] += 0.5
            return (
                self._project(
                    random_q,
                    mask,
                    fill_budget=True,
                    caps=caps,
                    budget=budget,
                    tie_break_priorities=tie_break_priorities,
                ),
                random_q,
            )
        self.online.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            mask_tensor = torch.as_tensor(
                mask, dtype=torch.bool, device=self.device
            ).unsqueeze(0)
            q = self.q_values_tensor(tensor, mask_tensor)[0].cpu().numpy()
        self.online.train()
        return (
            self._project(
                q, mask, caps=caps, budget=budget,
                tie_break_priorities=tie_break_priorities,
            ),
            q,
        )

    def store(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        mask,
        next_mask,
        caps=None,
        next_caps=None,
    ):
        if caps is None:
            caps = np.full(len(state), self.cfg.actions - 1, dtype=np.int64)
        if next_caps is None:
            next_caps = np.full(
                len(state), self.cfg.actions - 1, dtype=np.int64
            )
        self.replay.push(
            (
                np.asarray(state, dtype=np.float32),
                np.asarray(action, dtype=np.int64),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
                np.asarray(mask, dtype=bool),
                np.asarray(next_mask, dtype=bool),
                np.asarray(caps, dtype=np.int64),
                np.asarray(next_caps, dtype=np.int64),
            )
        )

    def _target_actions(self, next_q, next_masks, next_caps):
        actions = []
        for q, mask, caps in zip(next_q, next_masks, next_caps):
            actions.append(self._project(q, mask, caps=caps))
        return np.asarray(actions, dtype=np.int64)

    def learn(self, beta=0.4):
        if len(self.replay) < max(self.cfg.warmup, self.cfg.batch_size):
            return None
        batch, indices, importance = self.replay.sample(
            self.cfg.batch_size, beta
        )
        states, actions, rewards, next_states, dones, masks, next_masks, caps, next_caps = zip(
            *batch
        )
        states = torch.as_tensor(
            np.stack(states), dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            np.stack(actions), dtype=torch.long, device=self.device
        )
        rewards = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            np.stack(next_states), dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            dones, dtype=torch.float32, device=self.device
        )
        masks_t = torch.as_tensor(
            np.stack(masks), dtype=torch.bool, device=self.device
        )
        next_masks_np = np.stack(next_masks)
        next_caps_np = np.stack(next_caps)
        weights = torch.as_tensor(
            importance, dtype=torch.float32, device=self.device
        )

        model_states = self._transform_state_tensor(states)
        model_next_states = self._transform_state_tensor(next_states)
        with self._autocast():
            log_probabilities = self.online(model_states, masks_t)
        gather_index = actions.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, 1, self.cfg.atoms
        )
        chosen_log = log_probabilities.gather(2, gather_index).squeeze(2)

        with torch.no_grad(), self._autocast():
            next_masks_t = torch.as_tensor(
                next_masks_np, dtype=torch.bool, device=self.device
            )
            next_q = self.online.q_values(
                model_next_states, next_masks_t
            ).cpu().numpy()
            next_actions_np = self._target_actions(
                next_q, next_masks_np, next_caps_np
            )
            next_actions = torch.as_tensor(
                next_actions_np, dtype=torch.long, device=self.device
            )
            target_log = self.target(model_next_states, next_masks_t)
            target_index = next_actions.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, 1, self.cfg.atoms
            )
            next_distribution = (
                target_log.gather(2, target_index).squeeze(2).exp()
            )

            support = self.online.support
            delta = (self.cfg.v_max - self.cfg.v_min) / (
                self.cfg.atoms - 1
            )
            tz = rewards[:, None, None] + (
                self.cfg.gamma
                * (1.0 - dones[:, None, None])
                * support[None, None, :]
            )
            tz = tz.clamp(self.cfg.v_min, self.cfg.v_max)
            b = (tz - self.cfg.v_min) / delta
            lower = b.floor().long()
            upper = b.ceil().long()
            projected = torch.zeros_like(next_distribution)
            projected.scatter_add_(
                2, lower, next_distribution * (upper.float() - b)
            )
            projected.scatter_add_(
                2, upper, next_distribution * (b - lower.float())
            )
            equal = lower == upper
            projected.scatter_add_(
                2, lower, next_distribution * equal.float()
            )

        branch_loss = -(projected * chosen_log).sum(dim=-1)
        valid = masks_t.float()
        per_sample = (branch_loss * valid).sum(dim=1) / valid.sum(
            dim=1
        ).clamp_min(1.0)
        c51_loss = (per_sample * weights).mean()
        current_all_q = (
            log_probabilities.exp() * self.online.support
        ).sum(dim=-1)
        concavity_loss = self._concavity_loss(current_all_q, masks_t)
        trajectory_loss = self._trajectory_order_loss(states, masks_t)
        caps_t = torch.as_tensor(
            np.stack(caps), dtype=torch.long, device=self.device
        )
        action_levels = torch.arange(
            self.cfg.actions, device=self.device
        ).view(1, 1, -1)
        feasible_actions = action_levels <= caps_t.unsqueeze(-1)
        chosen_q = current_all_q.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        supervised_margin = (
            action_levels != actions.unsqueeze(-1)
        ).to(current_all_q.dtype) * float(self.cfg.demonstration_margin)
        margin_candidates = (current_all_q + supervised_margin).masked_fill(
            ~feasible_actions, -torch.inf
        )
        branch_margin_loss = torch.relu(
            margin_candidates.max(dim=-1).values - chosen_q
        )
        demonstration_per_sample = (
            (branch_margin_loss * valid).sum(dim=1)
            / valid.sum(dim=1).clamp_min(1.0)
        )
        demonstration_margin_loss = (
            demonstration_per_sample * weights
        ).mean()
        loss = (
            c51_loss
            + float(self.cfg.concavity_loss_weight) * concavity_loss
            + float(self.cfg.trajectory_loss_weight) * trajectory_loss
            + float(self.cfg.demonstration_margin_loss_weight)
            * demonstration_margin_loss
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        with torch.no_grad():
            current_q = (
                chosen_log.exp() * self.online.support[None, None, :]
            ).sum(dim=-1)
            target_q = (
                projected * self.online.support[None, None, :]
            ).sum(dim=-1)
            td = (
                (torch.abs(target_q - current_q) * valid).sum(dim=1)
                / valid.sum(dim=1).clamp_min(1.0)
            )
        self.replay.update(indices, td.cpu().numpy())
        self.train_steps += 1
        if self.train_steps % self.cfg.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        self.last_loss_terms = {
            "c51": float(c51_loss.item()),
            "trajectory_order": float(trajectory_loss.item()),
            "concavity": float(concavity_loss.item()),
            "demonstration_margin": float(demonstration_margin_loss.item()),
            "total": float(loss.item()),
        }
        return float(loss.item())

    def save(self, path, metadata):
        torch.save(
            {
                "online_state_dict": self.online.state_dict(),
                "target_state_dict": self.target.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.cfg.__dict__,
                "metadata": metadata,
            },
            path,
        )
