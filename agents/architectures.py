"""Network architectures for the HTA-MAC branching and independence ablation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalBranchingDuelingC51(nn.Module):
    """Tavakoli-style shared decision module and node-specific action heads."""

    def __init__(
        self,
        input_dim=50,
        hidden_dim=128,
        actions=4,
        atoms=51,
        v_min=-30.0,
        v_max=30.0,
        max_branches=100,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.actions = int(actions)
        self.atoms = int(atoms)
        self.max_branches = int(max_branches)
        self.register_buffer("support", torch.linspace(v_min, v_max, atoms))
        global_dim = self.max_branches * self.input_dim + self.max_branches
        self.shared_decision = nn.Sequential(
            nn.LayerNorm(global_dim),
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, atoms)
        )
        self.advantages = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, actions * atoms),
            )
            for _ in range(self.max_branches)
        )

    def _global_input(self, state, mask=None):
        if state.ndim != 3:
            raise ValueError("state must have shape [batch, branches, features]")
        batch, branches, features = state.shape
        if features != self.input_dim or branches > self.max_branches:
            raise ValueError("state dimensions exceed configured branch layout")
        if mask is None:
            mask = torch.ones(
                (batch, branches), dtype=torch.bool, device=state.device
            )
        if mask.shape != (batch, branches):
            raise ValueError("mask shape mismatch")
        padded_state = state.new_zeros(
            (batch, self.max_branches, self.input_dim)
        )
        padded_mask = state.new_zeros((batch, self.max_branches))
        padded_state[:, :branches] = state * mask.unsqueeze(-1)
        padded_mask[:, :branches] = mask
        return torch.cat((padded_state.flatten(1), padded_mask), dim=1), branches

    def forward(self, state, mask=None):
        global_input, branches = self._global_input(state, mask)
        hidden = self.shared_decision(global_input)
        value = self.value(hidden).view(-1, 1, 1, self.atoms)
        advantage = torch.stack(
            [
                head(hidden).view(-1, self.actions, self.atoms)
                for head in self.advantages[:branches]
            ],
            dim=1,
        )
        logits = value + advantage - advantage.mean(dim=2, keepdim=True)
        return F.log_softmax(logits, dim=-1)

    def q_values(self, state, mask=None):
        log_probabilities = self.forward(state, mask)
        return (log_probabilities.exp() * self.support).sum(dim=-1)


class EquivariantSetBranchingC51(nn.Module):
    """Permutation-equivariant C51 branches with invariant set context.

    Every physical node is processed by the same encoder and advantage head.
    Masked mean/max pooling provides global cluster context without flattening
    node order. Learned parameter count is independent of branch capacity.
    """

    def __init__(
        self,
        input_dim=58,
        hidden_dim=128,
        actions=4,
        atoms=51,
        v_min=-30.0,
        v_max=30.0,
        budget=12,
        max_branches=100,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.actions = int(actions)
        self.atoms = int(atoms)
        self.max_branches = int(max_branches)
        self.register_buffer("support", torch.linspace(v_min, v_max, atoms))
        self.register_buffer(
            "normalized_budget",
            torch.tensor(float(budget) / max(1, self.max_branches)),
        )
        self.node_encoder = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.global_context = nn.Sequential(
            nn.Linear(2 * self.hidden_dim + 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.atoms),
        )
        self.advantage = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.actions * self.atoms),
        )

    def _validate(self, state, mask):
        if state.ndim != 3:
            raise ValueError("state must have shape [batch, branches, features]")
        batch, branches, features = state.shape
        if features != self.input_dim or branches > self.max_branches:
            raise ValueError("state dimensions exceed configured set layout")
        if mask is None:
            mask = torch.ones(
                (batch, branches), dtype=torch.bool, device=state.device
            )
        if mask.shape != (batch, branches):
            raise ValueError("mask shape mismatch")
        return mask

    def _set_context(self, hidden, mask):
        valid = mask.unsqueeze(-1)
        masked = hidden * valid
        count = valid.sum(dim=1).clamp_min(1)
        # Accumulate in float64 so a node permutation cannot cross the strict
        # 1e-6 float32 equivariance gate solely through reduction order.
        mean = (
            masked.to(torch.float64).sum(dim=1)
            / count.to(torch.float64)
        ).to(hidden.dtype)
        negative_infinity = torch.finfo(hidden.dtype).min
        maximum = hidden.masked_fill(~valid, negative_infinity).max(dim=1).values
        any_valid = mask.any(dim=1, keepdim=True)
        maximum = torch.where(any_valid, maximum, torch.zeros_like(maximum))
        active_fraction = mask.to(hidden.dtype).mean(dim=1, keepdim=True)
        budget = self.normalized_budget.to(hidden.dtype).expand(
            hidden.shape[0], 1
        )
        return self.global_context(
            torch.cat((mean, maximum, active_fraction, budget), dim=-1)
        )

    def forward(self, state, mask=None):
        mask = self._validate(state, mask)
        hidden = self.node_encoder(state) * mask.unsqueeze(-1)
        context = self._set_context(hidden, mask)
        repeated_context = context.unsqueeze(1).expand(
            -1, state.shape[1], -1
        )
        value = self.value(context).view(-1, 1, 1, self.atoms)
        advantage = self.advantage(
            torch.cat((hidden, repeated_context), dim=-1)
        ).view(-1, state.shape[1], self.actions, self.atoms)
        logits = value + advantage - advantage.mean(dim=2, keepdim=True)
        log_probabilities = F.log_softmax(logits, dim=-1)
        uniform_log = -torch.log(
            state.new_tensor(float(self.atoms))
        )
        return torch.where(
            mask[:, :, None, None], log_probabilities, uniform_log
        )

    def q_values(self, state, mask=None):
        log_probabilities = self.forward(state, mask)
        return (log_probabilities.exp() * self.support).sum(dim=-1)


class _LocalDuelingC51(nn.Module):
    def __init__(self, input_dim, hidden_dim, actions, atoms):
        super().__init__()
        self.actions = actions
        self.atoms = atoms
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
        value = self.value(hidden).unsqueeze(1)
        advantage = self.advantage(hidden).view(-1, self.actions, self.atoms)
        logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        return F.log_softmax(logits, dim=-1)


class IndependentDuelingC51(nn.Module):
    """Ablation with one separately parameterized local DQN per node."""

    def __init__(
        self,
        input_dim=50,
        hidden_dim=128,
        actions=4,
        atoms=51,
        v_min=-30.0,
        v_max=30.0,
        max_branches=100,
    ):
        super().__init__()
        self.atoms = int(atoms)
        self.max_branches = int(max_branches)
        self.register_buffer("support", torch.linspace(v_min, v_max, atoms))
        self.networks = nn.ModuleList(
            _LocalDuelingC51(input_dim, hidden_dim, actions, atoms)
            for _ in range(self.max_branches)
        )

    def forward(self, state, mask=None):
        if state.ndim != 3 or state.shape[1] > self.max_branches:
            raise ValueError("state branch dimension exceeds independent layout")
        return torch.stack(
            [
                self.networks[index](state[:, index])
                for index in range(state.shape[1])
            ],
            dim=1,
        )

    def q_values(self, state, mask=None):
        log_probabilities = self.forward(state, mask)
        return (log_probabilities.exp() * self.support).sum(dim=-1)