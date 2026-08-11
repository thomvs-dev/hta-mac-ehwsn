"""Lock the architecture and side-study decisions before repaired training."""

from __future__ import annotations

import json
from pathlib import Path

from agents.architectures import EquivariantSetBranchingC51
from agents.branching_dqn import BranchingAgentConfig, BranchingDQNAgent


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config" / "paper_aligned_hasani2025_b16_qos_repaired.json"
DECISION = ROOT / "config" / "paper_aligned_hasani2025_architecture_decision_repaired.json"


def test_paper_aligned_b16_is_explicitly_a_secondary_side_study():
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    assert profile["track_role"] == "secondary_literature_alignment_side_study"
    assert profile["primary_track_replaced"] is False
    assert profile["primary_contributions_evaluated_by_this_profile"] is False
    assert set(profile["disabled_primary_contributions"]) == {
        "idle_listening_energy_accounting_C3",
        "hybrid_solar_thermal_harvesting_C1",
    }


def test_frozen_decision_keeps_flattened_node_head_architecture_retired():
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["selected_architecture_key"] == "equivariant_set_branching"
    assert decision["selected_python_class"].endswith("EquivariantSetBranchingC51")
    assert decision["rejected_python_class"].endswith("GlobalBranchingDuelingC51")
    assert decision["observation_contract"]["rank_or_percentile_features_present"] is False
    assert decision["mechanism_losses"]["trajectory_loss_weight"] == 1.0
    assert decision["mechanism_losses"]["concavity_loss_weight"] == 0.1
    assert decision["current_code_preflight"]["status"] == "gate_pass"


def test_checkpoint_producing_architecture_key_builds_equivariant_network():
    config = BranchingAgentConfig(
        input_dim=58,
        budget=16,
        max_branches=100,
        architecture="equivariant_set_branching",
        state_schema="phase2d_ttl_cap_v2",
        trajectory_loss_weight=1.0,
        concavity_loss_weight=0.1,
    )
    agent = BranchingDQNAgent(config, device="cpu")
    assert isinstance(agent.online, EquivariantSetBranchingC51)
    assert not hasattr(agent.online, "advantages")
