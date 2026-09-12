import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.sweep_step3_primary_branchwise_distillation import branchwise_loss


class _TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.values = torch.nn.Parameter(
            torch.tensor([0.0, 0.2, 0.1, -0.1], dtype=torch.float32)
        )

    def q_values(self, states, masks):
        return self.values.view(1, 1, 4).expand(states.shape[0], states.shape[1], 4)


class _TinyAgent:
    def __init__(self):
        self.online = _TinyPolicy()
        self.device = torch.device("cpu")
        self.cfg = SimpleNamespace(actions=4)

    @staticmethod
    def _transform_state_tensor(states):
        return states


def test_branchwise_loss_is_finite_balanced_and_preserves_source_policy():
    agent = _TinyAgent()
    source = copy.deepcopy(agent)
    states = torch.zeros((1, 4, 2))
    source_actions = torch.tensor([[1, 2, 1, 1]])
    targets = torch.tensor([[2, 1, 1, 1]])
    masks = torch.ones((1, 4), dtype=torch.bool)
    caps = torch.full((1, 4), 3)
    loss, changed, preservation = branchwise_loss(
        agent, source, states, source_actions, targets, masks, caps,
        temperature=0.05, changed_weight=1.0, preservation_weight=2.0,
    )
    assert torch.isfinite(loss)
    assert changed.item() > 0.0
    assert abs(preservation.item()) < 1e-7
    loss.backward()
    assert agent.online.values.grad is not None
    assert torch.isfinite(agent.online.values.grad).all()


def test_branchwise_contract_keeps_confirmation_seeds_prohibited():
    root = Path(__file__).resolve().parents[1]
    contract = json.loads(
        (root / "config" / "step3_primary_branchwise_distillation_sweep_v1.json").read_text()
    )
    assert contract["status"] == "frozen_before_primary_branchwise_sweep"
    assert not set(contract["development_seeds"]).intersection(contract["prohibited_seeds"])
    assert set(range(3400, 3405)).issubset(contract["prohibited_seeds"])
    assert contract["parallel_candidates"] * contract["threads_per_candidate"] == 16
