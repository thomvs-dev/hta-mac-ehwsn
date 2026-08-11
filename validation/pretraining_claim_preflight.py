"""Fail-fast claim-alignment gates to run before an expensive training sweep.

The input episode log is expected to come from a cheap representative probe or
smoke run produced by the intended training environment.  This command also
reruns the Phase 2D foundation audit with the current source code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-jsonl", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--environment-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-seeds", default="2400,2401,2402,2403,2404")
    parser.add_argument("--foundation-audit-seed", type=int, default=7399)
    parser.add_argument("--foundation-max-steps", type=int, default=5)
    parser.add_argument("--required-reward-term", default="deaths")
    parser.add_argument("--headline-event-field", default="t_fnd")
    parser.add_argument("--minimum-nonzero-term-records", type=int, default=1)
    parser.add_argument("--minimum-event-records", type=int, default=1)
    parser.add_argument("--minimum-training-horizon", type=int, required=True)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate_episode_gates(
    rows: list[dict],
    *,
    reward_term: str,
    headline_event_field: str,
    minimum_nonzero_term_records: int,
    minimum_event_records: int,
    minimum_training_horizon: int,
) -> dict:
    if not rows:
        raise ValueError("episode log is empty")
    reward_values = [
        float(row.get("raw_terms", {}).get(reward_term, 0.0)) for row in rows
    ]
    nonzero_term_records = sum(abs(value) > 0.0 for value in reward_values)
    event_records = sum(row.get(headline_event_field) is not None for row in rows)
    observed_horizon = max(int(row.get("steps", 0)) for row in rows)

    accounting_rows = []
    contract_rows = []
    missing_accounting = 0
    for row in rows:
        qos = row.get("qos_constraint")
        if not isinstance(qos, dict):
            missing_accounting += 1
            continue
        counts = qos.get("cumulative_counts", {})
        contract = qos.get("metric_contract", {})
        if "delivered" not in counts or "demand" not in counts:
            missing_accounting += 1
            continue
        accounting_rows.append(int(counts["delivered"]) <= int(counts["demand"]))
        contract_rows.append(
            contract.get("ratio_scope")
            == "episode_cumulative_target_backlog_service"
            and contract.get("demand_field") == "target_packets_offered"
            and contract.get("fairness_metric_name")
            == "target_cluster_service_fairness"
        )

    gates = {
        "claim_reward_term_activated": {
            "pass": nonzero_term_records >= minimum_nonzero_term_records,
            "term": reward_term,
            "nonzero_records": nonzero_term_records,
            "required_nonzero_records": minimum_nonzero_term_records,
            "maximum_absolute_value": max(map(abs, reward_values), default=0.0),
        },
        "accounting_invariant_all_records": {
            "pass": (
                missing_accounting == 0
                and len(accounting_rows) == len(rows)
                and all(accounting_rows)
                and all(contract_rows)
            ),
            "records": len(rows),
            "records_with_accounting": len(accounting_rows),
            "missing_accounting_records": missing_accounting,
            "delivered_le_demand_records": sum(accounting_rows),
            "contract_match_records": sum(contract_rows),
        },
        "headline_event_observed": {
            "pass": event_records >= minimum_event_records,
            "field": headline_event_field,
            "event_records": event_records,
            "required_event_records": minimum_event_records,
        },
        "training_horizon_covers_claim_event": {
            "pass": observed_horizon >= minimum_training_horizon,
            "observed_max_horizon": observed_horizon,
            "required_minimum_horizon": minimum_training_horizon,
        },
    }
    return gates


def main() -> None:
    args = parse_args()
    episodes_path = resolve(args.episodes_jsonl)
    checkpoint = resolve(args.checkpoint)
    profile = resolve(args.environment_profile)
    output = resolve(args.output)
    rows = load_jsonl(episodes_path)
    gates = evaluate_episode_gates(
        rows,
        reward_term=args.required_reward_term,
        headline_event_field=args.headline_event_field,
        minimum_nonzero_term_records=args.minimum_nonzero_term_records,
        minimum_event_records=args.minimum_event_records,
        minimum_training_horizon=args.minimum_training_horizon,
    )

    foundation_output = output.with_name(output.stem + ".foundation.json")
    foundation_output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-B",
        str(ROOT / "experiments" / "audit_phase2d_foundation.py"),
        str(checkpoint),
        "--output",
        str(foundation_output),
        "--max-steps",
        str(args.foundation_max_steps),
        "--development-seeds",
        args.development_seeds,
        "--environment-profile",
        str(profile),
        "--audit-seed",
        str(args.foundation_audit_seed),
    ]
    foundation_run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    foundation_payload = None
    if foundation_output.is_file():
        foundation_payload = json.loads(foundation_output.read_text(encoding="utf-8"))
    checkpoint_hash = sha256(checkpoint)
    foundation_pass = bool(
        foundation_run.returncode == 0
        and foundation_payload is not None
        and foundation_payload.get("status") == "gate_pass"
        and str(foundation_payload.get("checkpoint_sha256", "")).lower()
        == checkpoint_hash
    )
    gates["current_code_permutation_foundation"] = {
        "pass": foundation_pass,
        "command": command,
        "returncode": foundation_run.returncode,
        "checkpoint_sha256": checkpoint_hash,
        "audit_checkpoint_sha256": (
            foundation_payload.get("checkpoint_sha256")
            if foundation_payload is not None
            else None
        ),
        "audit_status": (
            foundation_payload.get("status")
            if foundation_payload is not None
            else None
        ),
        "stdout": foundation_run.stdout,
        "stderr": foundation_run.stderr,
    }

    overall_pass = all(item["pass"] for item in gates.values())
    payload = {
        "schema_version": 1,
        "status": "pretraining_claim_preflight_pass" if overall_pass else "pretraining_claim_preflight_fail",
        "stop_training_if_fail": True,
        "episodes_jsonl": str(episodes_path),
        "episodes_sha256": sha256(episodes_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "environment_profile": str(profile),
        "environment_profile_sha256": sha256(profile),
        "gates": gates,
        "overall_pass": overall_pass,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(output)}))
    if not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
