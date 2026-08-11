"""Combine cheap mechanism evidence, exact runtime, and strict architecture audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime_fingerprint import make_runtime_contract


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--environment-profile", type=Path, required=True)
    parser.add_argument("--ch-risk-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-contract-output", type=Path, required=True)
    parser.add_argument("--development-seeds", default="2400,2401,2402,2403,2404")
    args = parser.parse_args()
    resolve = lambda p: p if p.is_absolute() else ROOT / p
    mechanism, checkpoint, profile, risk, output, contract_path = map(resolve, (
        args.mechanism_report, args.checkpoint, args.environment_profile,
        args.ch_risk_config, args.output, args.runtime_contract_output,
    ))
    mechanism_payload = json.loads(mechanism.read_text())
    gates = {
        "deterministic_mechanism_probe": {
            "pass": mechanism_payload.get("status") == "step3_mechanism_probe_pass"
                    and mechanism_payload.get("learning_performed") is False,
            "status": mechanism_payload.get("status"),
        }
    }
    foundation_output = output.with_suffix(".foundation.json")
    command = [sys.executable, "-B", str(ROOT / "experiments/audit_phase2d_foundation.py"),
               str(checkpoint), "--output", str(foundation_output), "--max-steps", "5",
               "--development-seeds", args.development_seeds,
               "--environment-profile", str(profile), "--audit-seed", "20260809"]
    run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    foundation = json.loads(foundation_output.read_text()) if foundation_output.is_file() else {}
    gates["current_code_permutation_foundation"] = {
        "pass": run.returncode == 0 and foundation.get("status") == "gate_pass",
        "returncode": run.returncode, "audit_status": foundation.get("status"),
        "stdout": run.stdout, "stderr": run.stderr,
    }
    contract = make_runtime_contract(require_cuda=True)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    passed = all(item["pass"] for item in gates.values())
    payload = {
        "schema_version": 2,
        "status": "step3_pretraining_claim_preflight_pass" if passed else "step3_pretraining_claim_preflight_fail",
        "overall_pass": passed,
        "stop_training_if_fail": True,
        "probe_training_episodes": 1,
        "mechanism_probe_learning_performed": False,
        "platform_contract": "probe_and_full_training_must_use_exact_same_runtime_fingerprint",
        "cross_platform_tolerance_validated": False,
        "runtime_fingerprint_sha256": contract["fingerprint_sha256"],
        "ch_risk_config_sha256": sha256(risk),
        "mechanism_report_sha256": sha256(mechanism),
        "gates": gates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(output)}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
