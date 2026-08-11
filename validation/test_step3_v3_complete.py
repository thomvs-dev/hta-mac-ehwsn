from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import experiments.train_step3_v3 as v3
import experiments.train_step3_v3_complete as complete
from agents.branching_dqn import BranchingAgentConfig
from envs.step3_policy_observation import STEP3_CH_CONTEXT_SCHEMA


ROOT = Path(__file__).resolve().parents[1]


def test_complete_greedy_reports_packets_per_joule(monkeypatch):
    def base(*_):
        return {"clusters": [{"global_throughput": 10}]}, None, None

    monkeypatch.setattr(complete, "_BASE_GREEDY", base)
    evaluation, _, _ = complete.complete_greedy(
        None, [SimpleNamespace(step3_energy_consumed_j=2.0)], None, None
    )
    assert evaluation["clusters"][0]["packets_per_joule"] == 5.0
    assert evaluation["mean_packets_per_joule"] == 5.0


def test_episode_500_alias_is_written_before_finalization():
    root = ROOT / "outputs" / "validation_artifacts" / f"step3_v3_complete_{uuid4().hex}"
    v3._EXPORT_DIR = root / "drive"
    v3._OPTIMIZER_SEED = 5599
    v3._RUN_NAME = "interruption_test"
    agent = v3.RecoveryExportAgent(
        BranchingAgentConfig(
            input_dim=65, actions=4, budget=16, max_branches=100,
            architecture="equivariant_set_branching",
            state_schema=STEP3_CH_CONTEXT_SCHEMA, embedding_start_dim=33,
        )
    )
    local = root / "local"
    local.mkdir(parents=True)
    (local / "episodes.jsonl").write_text('{"episode":500}\n')
    complete.failure_safe_save(agent, local / "stability_episode_500.pt", {"episode": 500})
    checkpoint = v3._EXPORT_DIR / "training_complete_weights.pt"
    sidecar = json.loads((v3._EXPORT_DIR / "training_complete_weights.json").read_text())
    assert checkpoint.is_file()
    assert sidecar["status"] == "episode_500_weights_persisted_before_final_evaluation"
    assert sidecar["checkpoint_sha256"] == v3.sha256(checkpoint)
