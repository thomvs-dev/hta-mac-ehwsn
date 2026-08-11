"""Merge per-seed FND identity audit artifacts into one checked result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_phase3_fnd_node_identity import summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [path if path.is_absolute() else ROOT / path for path in args.inputs]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    reference = payloads[0]
    invariant_fields = (
        "checkpoint_sha256",
        "environment_profile_sha256",
        "run_phase3_pilot_sha256",
        "policies",
        "horizon",
    )
    for payload in payloads[1:]:
        for field in invariant_fields:
            if payload[field] != reference[field]:
                raise ValueError(f"cannot merge mismatched field {field}")
    records = [record for payload in payloads for record in payload["records"]]
    replay_checks = [
        row
        for payload in payloads
        for row in payload["archived_fnd_replay_checks"]
    ]
    seeds = sorted({int(record["seed"]) for record in records})
    policies = list(reference["policies"])
    expected = len(seeds) * len(policies)
    keys = {(record["seed"], record["policy"]) for record in records}
    if len(records) != expected or len(keys) != expected:
        raise ValueError("duplicate or missing seed-policy record")
    result = {
        "schema_version": 1,
        "status": "development_fnd_node_identity_audit_complete",
        "interpretation": reference["interpretation"],
        "checkpoint": reference["checkpoint"],
        "checkpoint_sha256": reference["checkpoint_sha256"],
        "environment_profile": reference["environment_profile"],
        "environment_profile_sha256": reference["environment_profile_sha256"],
        "run_phase3_pilot_sha256": reference["run_phase3_pilot_sha256"],
        "seeds": seeds,
        "policies": policies,
        "horizon": reference["horizon"],
        "held_out_or_confirmation_seeds_used": False,
        "merged_from": [str(path) for path in paths],
        "archived_fnd_replay_checks": replay_checks,
        "all_archived_fnd_rounds_reproduced": all(
            row["match"] for row in replay_checks
        ),
        "summary": summarize(records, policies),
        "records": sorted(records, key=lambda row: (row["seed"], row["policy"])),
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"output={output}")


if __name__ == "__main__":
    main()
