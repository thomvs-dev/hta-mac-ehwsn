"""Variable-cardinality Branching Dueling C51 agent for HTA-MAC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, state):
        hidden = self.trunk(state)
        value = self.value(hidden).unsqueeze(2)
        advantage = self.advantage(hidden).view(
            *hidden.shape[:-1], self.actions, self.atoms
        )
        logits = value + advantage - advantage.mean(dim=2, keepdim=True)
        return F.log_softmax(logits, dim=-1)

    def q_values(self, state):
        log_probabilities = self.forward(state)
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


class BranchingDQNAgent:
    def __init__(self, config: BranchingAgentConfig, device="cpu"):
        self.cfg = config
        self.device = torch.device(device)
        kwargs = dict(
            input_dim=config.input_dim,
            actions=config.actions,
            atoms=config.atoms,
            v_min=config.v_min,
            v_max=config.v_max,
        )
        self.online = BranchingDuelingC51(**kwargs).to(self.device)
        self.target = BranchingDuelingC51(**kwargs).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=config.learning_rate
        )
        self.replay = PrioritizedReplay(config.replay_capacity)
        self.train_steps = 0

    def _project(
        self, q_values, mask, *, fill_budget=False, caps=None, budget=None
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
        )
        allocation = np.minimum(allocation, caps)
        allocation[~mask] = 0
        return allocation

    def act(self, state, mask, epsilon=0.0, caps=None, budget=None):
        if np.random.random() < epsilon:
            random_q = np.random.normal(
                size=(len(state), self.cfg.actions)
            )
            random_q[:, 1:] += 0.5
            return (
                self._project(
                    random_q, mask, fill_budget=True, caps=caps, budget=budget
                ),
                random_q,
            )
        self.online.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q = self.online.q_values(tensor)[0].cpu().numpy()
        self.online.train()
        return self._project(q, mask, caps=caps, budget=budget), q

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

        log_probabilities = self.online(states)
        gather_index = actions.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, 1, self.cfg.atoms
        )
        chosen_log = log_probabilities.gather(2, gather_index).squeeze(2)

        with torch.no_grad():
            next_q = self.online.q_values(next_states).cpu().numpy()
            next_actions_np = self._target_actions(
                next_q, next_masks_np, next_caps_np
            )
            next_actions = torch.as_tensor(
                next_actions_np, dtype=torch.long, device=self.device
            )
            target_log = self.target(next_states)
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
        loss = (per_sample * weights).mean()
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
