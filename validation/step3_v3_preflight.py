"""Freeze Step 3 v3 schema, runtime, mechanism, and foundation gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.train_phase2_dynamic_curriculum as trainer
from agents.ch_depletion_risk import validate_ch_risk_config
from agents.qos_constraints_v3 import Step3QoSConstraintConfig
from core.runtime_fingerprint import make_runtime_contract
from envs.step3_lifetime_env import RoleSeparatedScheduledMACEnv, configure_step3_risk
from envs.step3_policy_observation import STEP3_CH_CONTEXT_SCHEMA
from envs.step3_v3_env import Step3V3DynamicClusterTrainingEnv


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--environment-profile", type=Path, required=True)
    parser.add_argument("--ch-risk-config", type=Path, required=True)
    parser.add_argument("--step3-qos-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-contract-output", type=Path, required=True)
    args = parser.parse_args()
    resolve = lambda p: p if p.is_absolute() else ROOT / p
    mechanism, checkpoint, profile, risk_path, qos_path, output, contract_path = map(resolve, (
        args.mechanism_report, args.checkpoint, args.environment_profile,
        args.ch_risk_config, args.step3_qos_config, args.output,
        args.runtime_contract_output,
    ))
    risk = validate_ch_risk_config(json.loads(risk_path.read_text()))
    Step3QoSConstraintConfig.from_payload(json.loads(qos_path.read_text()))
    mechanism_payload = json.loads(mechanism.read_text())
    configure_step3_risk(risk)
    trainer.ScheduledIntraClusterMACEnv = RoleSeparatedScheduledMACEnv
    trainer.DynamicClusterTrainingEnv = Step3V3DynamicClusterTrainingEnv
    environments, manifest, _ = trainer.build_curriculum(
        [2400], 5, observation_schema=STEP3_CH_CONTEXT_SCHEMA,
        environment_profile=profile,
    )
    env = environments[0]
    observation, mask, _ = env.reset()
    layout = env.observation_layout
    context_start = int(layout["embedding_start"]) - int(layout["scheduled_ch_context_features"])
    reserve_index = context_start
    before = observation[:, reserve_index].copy()
    env.base.energy[int(env.ch)] *= 0.5
    after = env._observation(env.base._state())[:, reserve_index]
    context_sensitive = bool(np.all(after < before) and np.allclose(after, before * 0.5, atol=1e-6))
    context_broadcast = bool(np.allclose(before, before[0], atol=0.0))
    gates = {
        "mechanism_probe": mechanism_payload.get("status") == "step3_mechanism_probe_pass",
        "schema_is_v3": env.observation_schema == STEP3_CH_CONTEXT_SCHEMA,
        "input_dimension_is_65": observation.shape == (100, 65),
        "embedding_start_is_33": layout["embedding_start"] == 33,
        "ch_context_broadcast_without_node_id": context_broadcast,
        "ch_reserve_is_observable": context_sensitive,
        "schedule_schema_unchanged": manifest[0]["schedule_schema_version"] == "paper_aligned_exogenous_leach_v1",
    }
    foundation_output = output.with_suffix(".foundation.json")
    command = [sys.executable, "-B", str(ROOT / "experiments/audit_step3_v3_foundation.py"),
               str(checkpoint), "--output", str(foundation_output), "--max-steps", "5",
               "--development-seeds", "2400,2401,2402,2403,2404",
               "--environment-profile", str(profile), "--audit-seed", "20260809"]
    run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    foundation = json.loads(foundation_output.read_text()) if foundation_output.is_file() else {}
    gates["same_platform_permutation_foundation"] = run.returncode == 0 and foundation.get("status") == "gate_pass"
    contract = make_runtime_contract(require_cuda=True)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    passed = all(gates.values())
    payload = {
        "schema_version": 1,
        "status": "step3_v3_pretraining_pass" if passed else "step3_v3_pretraining_fail",
        "overall_pass": passed,
        "stop_training_if_fail": True,
        "observation_schema": STEP3_CH_CONTEXT_SCHEMA,
        "runtime_fingerprint_sha256": contract["fingerprint_sha256"],
        "ch_risk_config_sha256": sha256(risk_path),
        "qos_config_sha256": sha256(qos_path),
        "mechanism_report_sha256": sha256(mechanism),
        "foundation_checkpoint_sha256": sha256(checkpoint),
        "cross_platform_tolerance_validated": False,
        "gates": gates,
        "foundation_stdout": run.stdout,
        "foundation_stderr": run.stderr,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "gates": gates}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
