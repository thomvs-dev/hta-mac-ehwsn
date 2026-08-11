"""Idempotently finalize a completed Step 3 v3 lineage without retraining."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agents.reward_model as reward_module
import experiments.recover_phase2_finalization as recovery
import experiments.train_phase2_dynamic_curriculum as trainer
import experiments.train_step3_v3 as v3
from agents.ch_depletion_risk import validate_ch_risk_config
from agents.qos_constraints_v3 import Step3QoSConstraintConfig
from core.runtime_fingerprint import validate_runtime_contract
from envs.step3_lifetime_env import RoleSeparatedScheduledMACEnv, configure_step3_risk
from envs.step3_policy_observation import STEP3_CH_CONTEXT_SCHEMA
from envs.step3_v3_env import Step3V3DynamicClusterTrainingEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--optimizer-seed", type=int, required=True)
    parser.add_argument("--training-git-hash", required=True)
    parser.add_argument("--ch-risk-config", type=Path, required=True)
    parser.add_argument("--step3-qos-config", type=Path, required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--environment-profile", type=Path, required=True)
    parser.add_argument("--checkpoint-export-dir", type=Path)
    parser.add_argument("--development-seeds", default="2400,2401,2402,2403,2404")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    resolve = lambda p: p if p.is_absolute() else ROOT / p
    risk_path, qos_path, contract_path, preflight_path, profile = map(resolve, (
        args.ch_risk_config, args.step3_qos_config, args.runtime_contract,
        args.preflight_report, args.environment_profile,
    ))
    risk = validate_ch_risk_config(json.loads(risk_path.read_text()))
    qos = Step3QoSConstraintConfig.from_payload(json.loads(qos_path.read_text()))
    contract = json.loads(contract_path.read_text())
    if not validate_runtime_contract(contract)["pass"]:
        raise RuntimeError("finalizer runtime differs from the frozen training runtime")
    preflight = json.loads(preflight_path.read_text())
    for key, expected in (
        ("status", "step3_v3_pretraining_pass"),
        ("runtime_fingerprint_sha256", contract["fingerprint_sha256"]),
        ("ch_risk_config_sha256", v3.sha256(risk_path)),
        ("qos_config_sha256", v3.sha256(qos_path)),
    ):
        if preflight.get(key) != expected:
            raise RuntimeError(f"finalizer preflight mismatch: {key}")
    configure_step3_risk(risk)
    trainer.ScheduledIntraClusterMACEnv = RoleSeparatedScheduledMACEnv
    trainer.DynamicClusterTrainingEnv = Step3V3DynamicClusterTrainingEnv
    trainer.PHASE2D_POLICY_SCHEMA = STEP3_CH_CONTEXT_SCHEMA
    trainer.TERM_ORDER = tuple(reward_module.TERM_ORDER) + ("ch_depletion_risk",)
    v3._RISK_CONFIG, v3._QOS_CONFIG = risk, qos
    v3._RUN_NAME, v3._OPTIMIZER_SEED = args.run_name, args.optimizer_seed
    v3._EXPORT_DIR = resolve(args.checkpoint_export_dir) if args.checkpoint_export_dir else None

    def build_v3(seeds, max_steps):
        return trainer.build_curriculum(
            seeds, max_steps, observation_schema=STEP3_CH_CONTEXT_SCHEMA,
            environment_profile=profile,
        )

    recovery.build_curriculum = build_v3
    recovery.RewardModel = v3.Step3V3RewardModel
    recovery.greedy_curriculum_evaluation = v3.v3_greedy_evaluation
    recovery.contribution_balance = v3.contribution_balance
    sys.argv = [
        sys.argv[0], "--run-name", args.run_name,
        "--optimizer-seed", str(args.optimizer_seed),
        "--training-git-hash", args.training_git_hash,
        "--development-seeds", args.development_seeds,
        "--episodes", str(args.episodes), "--max-steps", str(args.max_steps),
        "--device", args.device,
    ]
    legacy_result = recovery.main()
    run_dir = ROOT / "outputs" / "phase2" / args.run_name
    result = v3.postprocess(run_dir, legacy_result, risk_path, qos_path, contract_path, preflight_path)
    recovery_payload = {
        "status": "finalization_recovered_without_retraining" if result in (0, 3) else "finalization_failed",
        "weights_retrained": False,
        "source_checkpoint": f"stability_episode_{args.episodes}.pt",
        "source_checkpoint_sha256": v3.sha256(run_dir / f"stability_episode_{args.episodes}.pt"),
        "final_checkpoint_sha256": v3.sha256(run_dir / "branching_c51.pt") if (run_dir / "branching_c51.pt").is_file() else None,
        "gate_exit_code": result,
    }
    path = run_dir / "step3_v3_finalization_recovery.json"
    path.write_text(json.dumps(recovery_payload, indent=2) + "\n")
    if v3._EXPORT_DIR is not None:
        v3.atomic_copy(path, Path(v3._EXPORT_DIR) / path.name)
    print(json.dumps(recovery_payload, indent=2))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
