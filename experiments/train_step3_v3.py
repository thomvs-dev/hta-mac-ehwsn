"""Train Step 3 v3 with observable CH context, complete gates, and Drive salvage."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agents.reward_model as reward_module
import experiments.train_phase2_dynamic_curriculum as trainer
from agents.branching_dqn import BranchingDQNAgent
from agents.ch_depletion_risk import validate_ch_risk_config
from agents.qos_constraints_v3 import Step3QoSConstraintConfig, Step3QoSConstraintController
from core.runtime_fingerprint import validate_runtime_contract
from envs.step3_lifetime_env import RoleSeparatedScheduledMACEnv, configure_step3_risk
from envs.step3_policy_observation import STEP3_CH_CONTEXT_SCHEMA
from envs.step3_v3_env import Step3V3DynamicClusterTrainingEnv


_RISK_CONFIG = None
_QOS_CONFIG = None
_EXPORT_DIR = None
_RUN_NAME = None
_OPTIMIZER_SEED = None
_ORIGINAL_GREEDY = trainer.greedy_curriculum_evaluation


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_copy(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


class Step3V3RewardModel:
    def __init__(self, scales, weights):
        self.scales, self.weights = dict(scales), dict(weights)
        self.scales["ch_depletion_risk"] = float(_RISK_CONFIG["scale"])
        self.weights["ch_depletion_risk"] = float(_RISK_CONFIG["weight"])

    def evaluate(self, raw_terms):
        weighted = {
            name: self.weights[name] * float(raw_terms[name]) / max(self.scales[name], 1e-12)
            for name in trainer.TERM_ORDER
        }
        return float(sum(weighted.values())), weighted


class RecoveryExportAgent(BranchingDQNAgent):
    def save(self, path, metadata=None):
        super().save(path, metadata)
        path = Path(path)
        if _EXPORT_DIR is None or not path.name.startswith("stability_episode_"):
            return
        destination = Path(_EXPORT_DIR) / path.name
        atomic_copy(path, destination)
        episodes = path.parent / "episodes.jsonl"
        if episodes.is_file():
            atomic_copy(episodes, Path(_EXPORT_DIR) / "episodes.jsonl")
        recovery = {
            "schema_version": 1,
            "status": "salvage_checkpoint_not_exact_midrun_resume",
            "optimizer_seed": _OPTIMIZER_SEED,
            "run_name": _RUN_NAME,
            "checkpoint": destination.name,
            "checkpoint_sha256": sha256(destination),
            "episode": (metadata or {}).get("episode"),
            "finalization_recoverable": (metadata or {}).get("episode") == 500,
        }
        sidecar = Path(_EXPORT_DIR) / (path.stem + ".recovery.json")
        temporary = sidecar.with_name(sidecar.name + ".partial")
        temporary.write_text(json.dumps(recovery, indent=2) + "\n")
        temporary.replace(sidecar)


def v3_greedy_evaluation(agent, environments, max_branches, reward_model):
    evaluation, first_env, first_observation = _ORIGINAL_GREEDY(
        agent, environments, max_branches, reward_model
    )
    rows = evaluation["clusters"]
    for row, env in zip(rows, environments):
        counts = env.step3_qos_counts
        demand = max(1, int(counts["demand"]))
        delivery = int(counts["delivered"]) / demand
        stale = int(counts["stale"]) / demand
        fairness = float(counts["fairness"])
        passed = (
            delivery >= _QOS_CONFIG.minimum_delivery_ratio
            and stale <= _QOS_CONFIG.maximum_stale_drop_ratio
            and fairness >= _QOS_CONFIG.minimum_queue_fairness
        )
        row["target_qos"] = {
            "delivery_ratio": delivery,
            "stale_ratio": stale,
            "fairness": fairness,
            "joint_pass": bool(passed),
            "delivered": int(counts["delivered"]),
            "demand": int(counts["demand"]),
            "stale": int(counts["stale"]),
        }
    joint = sum(row["target_qos"]["joint_pass"] for row in rows)
    evaluation["step3_target_qos"] = {
        "pairs": len(rows),
        "joint_pass_count": joint,
        "joint_pass_fraction": joint / max(1, len(rows)),
        "required_fraction": 0.90,
        "pass": joint / max(1, len(rows)) >= 0.90,
        "thresholds": {
            "delivery_min": _QOS_CONFIG.minimum_delivery_ratio,
            "stale_max": _QOS_CONFIG.maximum_stale_drop_ratio,
            "fairness_min": _QOS_CONFIG.minimum_queue_fairness,
        },
    }
    return evaluation, first_env, first_observation


def contribution_balance(rows):
    totals = {
        name: sum(abs(float(row["weighted_terms"].get(name, 0.0))) for row in rows)
        for name in trainer.TERM_ORDER
    }
    total = sum(totals.values())
    fractions = {name: value / total if total else 0.0 for name, value in totals.items()}
    dominant = max(fractions, key=fractions.get) if total else None
    return totals, fractions, dominant


def postprocess(run_dir: Path, legacy_result: int, risk_path, qos_path, contract_path, preflight_path):
    summary_path, episodes_path = run_dir / "summary.json", run_dir / "episodes.jsonl"
    if not summary_path.is_file() or not episodes_path.is_file():
        return 4
    summary = json.loads(summary_path.read_text())
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    tail = rows[-min(50, len(rows)):]
    totals, fractions, dominant = contribution_balance(tail)
    risk_fraction = fractions.get("ch_depletion_risk", 0.0)
    risk_by_seed = {
        seed: sum(abs(float(row["raw_terms"].get("ch_depletion_risk", 0.0))) for row in rows if int(row["seed"]) == seed)
        for seed in sorted({int(row["seed"]) for row in rows})
    }
    active_penalty_fractions = []
    for row in tail:
        qos = row.get("qos_constraint") or {}
        violation = float((qos.get("positive_violations") or {}).get("delivery", 0.0))
        if violation > 0:
            penalty = abs(float(row.get("constraint_penalty_total", 0.0)))
            physical = abs(float(row.get("raw_physical_reward", 0.0)))
            active_penalty_fractions.append(penalty / max(penalty + physical, 1e-12))
    penalty_fraction = float(np.mean(active_penalty_fractions)) if active_penalty_fractions else 0.0
    snapshots = summary.get("policy_stability_snapshots", [])[-3:]
    snapshot_qos = [item["evaluation"].get("step3_target_qos", {}) for item in snapshots]
    qos_gate = len(snapshot_qos) == 3 and all(item.get("pass") is True for item in snapshot_qos)
    risk_gate = all(value > 0.0 for value in risk_by_seed.values()) and risk_fraction < float(_RISK_CONFIG["max_allowed_absolute_reward_fraction"])
    penalty_gate = (
        _QOS_CONFIG.target_penalty_fraction_min <= penalty_fraction
        <= _QOS_CONFIG.target_penalty_fraction_max
    )
    legacy_gate = bool(summary.get("phase2_curriculum_gate_pass")) and legacy_result == 0
    v3_gate = bool(legacy_gate and qos_gate and risk_gate and penalty_gate)
    summary["legacy_phase2_curriculum_gate_pass"] = bool(summary.get("phase2_curriculum_gate_pass"))
    summary["phase2_curriculum_gate_pass"] = v3_gate
    summary["status"] = "pass" if v3_gate else "fail"
    summary["reward_balance"] = {
        "last_n_episodes": len(tail), "absolute_totals": totals,
        "fractions": fractions, "dominant_term": dominant,
        "pathological_domination": bool(max(fractions.values(), default=0.0) > 0.80),
        "threshold": 0.80,
    }
    summary["step3_v3"] = {
        "status": "development_candidate" if v3_gate else "development_gate_fail",
        "observation_schema": STEP3_CH_CONTEXT_SCHEMA,
        "scheduled_ch_context_observable": True,
        "ch_schedule_modified": False,
        "learned_intervention": "mac_allocation_only",
        "risk_config_sha256": sha256(risk_path),
        "qos_config_sha256": sha256(qos_path),
        "runtime_contract_sha256": sha256(contract_path),
        "preflight_sha256": sha256(preflight_path),
        "risk_active_by_seed": risk_by_seed,
        "risk_tail_absolute_reward_fraction": risk_fraction,
        "risk_non_dominating_pass": risk_gate,
        "active_delivery_penalty_fraction_mean": penalty_fraction,
        "active_delivery_penalty_fraction_target": [
            _QOS_CONFIG.target_penalty_fraction_min,
            _QOS_CONFIG.target_penalty_fraction_max,
        ],
        "penalty_geometry_pass": penalty_gate,
        "final_three_snapshot_target_qos": snapshot_qos,
        "target_qos_gate_pass": qos_gate,
        "overall_gate_pass": v3_gate,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    checkpoint_path = run_dir / "branching_c51.pt"
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint["metadata"] = summary
        torch.save(checkpoint, checkpoint_path)
    if _EXPORT_DIR is not None:
        for path in (summary_path, checkpoint_path, episodes_path):
            if path.is_file():
                atomic_copy(path, Path(_EXPORT_DIR) / path.name)
    return 0 if v3_gate else 3


def main():
    global _RISK_CONFIG, _QOS_CONFIG, _EXPORT_DIR, _RUN_NAME, _OPTIMIZER_SEED
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ch-risk-config", type=Path, required=True)
    parser.add_argument("--step3-qos-config", type=Path, required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--checkpoint-export-dir", type=Path)
    known, remaining = parser.parse_known_args()
    resolve = lambda p: p if p.is_absolute() else ROOT / p
    risk_path, qos_path, contract_path, preflight_path = map(resolve, (
        known.ch_risk_config, known.step3_qos_config,
        known.runtime_contract, known.preflight_report,
    ))
    _RISK_CONFIG = validate_ch_risk_config(json.loads(risk_path.read_text()))
    _QOS_CONFIG = Step3QoSConstraintConfig.from_payload(json.loads(qos_path.read_text()))
    contract = json.loads(contract_path.read_text())
    runtime = validate_runtime_contract(contract)
    if not runtime["pass"]:
        raise RuntimeError(f"probe/training runtime mismatch: {runtime}")
    preflight = json.loads(preflight_path.read_text())
    if preflight.get("status") != "step3_v3_pretraining_pass":
        raise RuntimeError("Step 3 v3 preflight has not passed")
    for field, expected in (
        ("runtime_fingerprint_sha256", contract["fingerprint_sha256"]),
        ("ch_risk_config_sha256", sha256(risk_path)),
        ("qos_config_sha256", sha256(qos_path)),
    ):
        if preflight.get(field) != expected:
            raise RuntimeError(f"preflight mismatch: {field}")
    _EXPORT_DIR = resolve(known.checkpoint_export_dir) if known.checkpoint_export_dir else None
    _RUN_NAME = remaining[remaining.index("--run-name") + 1]
    _OPTIMIZER_SEED = int(remaining[remaining.index("--optimizer-seed") + 1])
    configure_step3_risk(_RISK_CONFIG)
    trainer.ScheduledIntraClusterMACEnv = RoleSeparatedScheduledMACEnv
    trainer.DynamicClusterTrainingEnv = Step3V3DynamicClusterTrainingEnv
    trainer.PHASE2D_POLICY_SCHEMA = STEP3_CH_CONTEXT_SCHEMA
    trainer.TERM_ORDER = tuple(reward_module.TERM_ORDER) + ("ch_depletion_risk",)
    trainer.RewardModel = Step3V3RewardModel
    trainer.QoSConstraintConfig = Step3QoSConstraintConfig
    trainer.QoSConstraintController = Step3QoSConstraintController
    trainer.BranchingDQNAgent = RecoveryExportAgent
    trainer.greedy_curriculum_evaluation = v3_greedy_evaluation
    # The underlying trainer still expects its normal QoS CLI name.
    remaining.extend(["--qos-constraint-config", str(qos_path)]) if "--qos-constraint-config" not in remaining else None
    sys.argv = [sys.argv[0], *remaining]
    legacy_result = trainer.main()
    return postprocess(ROOT / "outputs" / "phase2" / _RUN_NAME, legacy_result, risk_path, qos_path, contract_path, preflight_path)


if __name__ == "__main__":
    raise SystemExit(main())
