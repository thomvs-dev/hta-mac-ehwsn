"""Hard Step 3 gate: claim activation, bounded risk, and exact runtime identity."""

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
from validation.pretraining_claim_preflight import evaluate_episode_gates, load_jsonl


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_step3_gates(rows: list[dict], risk_config: dict, minimum_horizon: int) -> dict:
    gates = evaluate_episode_gates(
        rows,
        reward_term="deaths",
        headline_event_field="t_fnd",
        minimum_nonzero_term_records=1,
        minimum_event_records=1,
        minimum_training_horizon=minimum_horizon,
    )
    risk_raw = [abs(float(row.get("raw_terms", {}).get("ch_depletion_risk", 0))) for row in rows]
    risk_weighted = [abs(float(row.get("weighted_terms", {}).get("ch_depletion_risk", 0))) for row in rows]
    total_weighted = [
        sum(abs(float(value)) for value in row.get("weighted_terms", {}).values())
        for row in rows
    ]
    fractions = [risk / max(total, 1e-12) for risk, total in zip(risk_weighted, total_weighted)]
    cap = float(risk_config["max_allowed_absolute_reward_fraction"])
    gates["ch_risk_activated"] = {
        "pass": sum(value > 0 for value in risk_raw) >= 1,
        "nonzero_records": sum(value > 0 for value in risk_raw),
    }
    gates["ch_risk_non_dominating"] = {
        "pass": bool(fractions) and max(fractions) <= cap,
        "maximum_absolute_reward_fraction": max(fractions, default=None),
        "frozen_cap": cap,
    }
    return gates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-jsonl", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--environment-profile", type=Path, required=True)
    parser.add_argument("--ch-risk-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-contract-output", type=Path, required=True)
    parser.add_argument("--minimum-training-horizon", type=int, default=1200)
    parser.add_argument("--development-seeds", default="2400,2401,2402,2403,2404")
    parser.add_argument("--foundation-audit-seed", type=int, default=7399)
    parser.add_argument("--foundation-max-steps", type=int, default=5)
    args = parser.parse_args()
    resolve = lambda p: p if p.is_absolute() else ROOT / p
    paths = {name: resolve(getattr(args, name)) for name in (
        "episodes_jsonl", "checkpoint", "environment_profile", "ch_risk_config", "output", "runtime_contract_output"
    )}
    risk_config = json.loads(paths["ch_risk_config"].read_text(encoding="utf-8"))
    rows = load_jsonl(paths["episodes_jsonl"])
    gates = evaluate_step3_gates(rows, risk_config, args.minimum_training_horizon)
    foundation_output = paths["output"].with_suffix(".foundation.json")
    command = [
        sys.executable, "-B", str(ROOT / "experiments" / "audit_phase2d_foundation.py"),
        str(paths["checkpoint"]), "--output", str(foundation_output), "--max-steps",
        str(args.foundation_max_steps), "--development-seeds", args.development_seeds,
        "--environment-profile", str(paths["environment_profile"]), "--audit-seed",
        str(args.foundation_audit_seed),
    ]
    run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    foundation = json.loads(foundation_output.read_text(encoding="utf-8")) if foundation_output.is_file() else {}
    gates["current_code_permutation_foundation"] = {
        "pass": run.returncode == 0 and foundation.get("status") == "gate_pass",
        "returncode": run.returncode,
        "audit_status": foundation.get("status"),
        "stdout": run.stdout,
        "stderr": run.stderr,
    }
    runtime_contract = make_runtime_contract(require_cuda=True)
    paths["runtime_contract_output"].parent.mkdir(parents=True, exist_ok=True)
    paths["runtime_contract_output"].write_text(json.dumps(runtime_contract, indent=2) + "\n", encoding="utf-8")
    overall = all(gate["pass"] for gate in gates.values())
    payload = {
        "schema_version": 1,
        "status": "step3_pretraining_claim_preflight_pass" if overall else "step3_pretraining_claim_preflight_fail",
        "overall_pass": overall,
        "stop_training_if_fail": True,
        "platform_contract": "probe_and_full_training_must_use_exact_same_runtime_fingerprint",
        "cross_platform_tolerance_validated": False,
        "runtime_fingerprint_sha256": runtime_contract["fingerprint_sha256"],
        "ch_risk_config_sha256": digest(paths["ch_risk_config"]),
        "gates": gates,
    }
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    paths["output"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(paths["output"])}))
    if not overall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
