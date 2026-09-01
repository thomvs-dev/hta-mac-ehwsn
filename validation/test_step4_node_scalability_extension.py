import json
import math
from pathlib import Path

import numpy as np
import pytest

from experiments.run_step4_node_scalability_extension import (
    capped_energy_proportional_action,
    load_capacity_agent,
    load_contract,
    scalability_schedule_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config" / "step4_node_scalability_50_300_v1.json"


def test_scalability_contract_is_frozen_and_complete():
    contract = load_contract(CONTRACT)
    assert [row["nodes"] for row in contract["scenarios"]] == [50, 100, 150, 200, 250, 300]
    assert contract["evaluation_seeds"] == list(range(3900, 3920))
    assert contract["decision_rules"]["all_sizes_reported"] is True
    assert contract["capacity_expansion"]["learned_weights_changed"] is False


def test_capacity_expansion_preserves_parameters_and_feasibility():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    agent, checkpoint, expanded = load_capacity_agent(
        ROOT / contract["source_checkpoint"], nodes=300, budget=24
    )
    assert expanded is True
    assert agent.cfg.max_branches == 300
    assert sum(parameter.numel() for parameter in agent.online.parameters()) == 116033
    assert np.isclose(float(agent.online.normalized_budget), 24 / 300)
    for key, tensor in checkpoint["online_state_dict"].items():
        if key != "normalized_budget":
            assert np.array_equal(
                tensor.detach().cpu().numpy(),
                agent.online.state_dict()[key].detach().cpu().numpy(),
            )
    state = np.zeros((300, agent.cfg.input_dim), dtype=np.float32)
    mask = np.zeros(300, dtype=bool); mask[:20] = True
    caps = np.zeros(300, dtype=np.int64); caps[:20] = 3
    action, q_values = agent.act(
        state, mask, epsilon=0.0, caps=caps, budget=24,
        tie_break_priorities=np.arange(300),
    )
    assert q_values.shape == (300, 4)
    assert action.shape == (300,)
    assert int(action.sum()) <= 24
    assert np.all(action <= caps)
    assert np.all(action[~mask] == 0)


@pytest.mark.parametrize("nodes", [150, 200, 250, 300])
def test_extended_schedule_has_exactly_balanced_ch_exposure(nodes):
    ch_count = max(1, int(round(nodes * 0.05)))
    period = nodes // math.gcd(nodes, ch_count)
    bundle = scalability_schedule_bundle(
        seed=3900, nodes=nodes, horizon=period, field_size=100.0,
        bs_position=(50.0, 175.0), solar_states=3, thermal_states=3,
    )
    counts = np.zeros(nodes, dtype=np.int64)
    for frame in bundle["schedule"]:
        counts[np.asarray(frame["cluster_heads"], dtype=np.int64)] += 1
    assert counts.min() == counts.max()
    assert bundle["schedule_metadata"]["exact_balance_period_rounds"] == period
    assert bundle["schedule"][0]["positions"] is bundle["schedule"][1]["positions"]
    assert bundle["schedule"][0]["stgcn_embedding"] is bundle["schedule"][1]["stgcn_embedding"]


def test_energy_proportional_baseline_obeys_queue_caps():
    class Base:
        n_nodes = 5
        initial_energy = np.ones(5)
        energy = np.asarray([1.0, 0.9, 0.8, 0.7, 0.6])
        class cfg:
            initial_energy_j = 1.0
    class Env:
        base = Base()
    mask = np.ones(5, dtype=bool)
    caps = np.asarray([0, 1, 2, 0, 3], dtype=np.int64)
    action = capped_energy_proportional_action(Env(), mask, caps, 24, 4.0)
    assert int(action.sum()) == int(caps.sum())
    assert np.all(action <= caps)
