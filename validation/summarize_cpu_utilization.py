"""Create a corrected utilization summary while preserving raw monitor evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
    samples = list(payload.get("samples", []))
    valid = [
        row for row in samples
        if 0.0 <= float(row["total_machine_cpu_percent"]) <= 100.0
    ]
    values = [float(row["total_machine_cpu_percent"]) for row in valid]
    result = {
        "schema_version": 1,
        "status": "corrected_monitor_summary_raw_preserved",
        "raw_input": str(args.input.resolve()),
        "raw_sample_count": len(samples),
        "valid_sample_count": len(valid),
        "excluded_sample_count": len(samples) - len(valid),
        "validity_rule": "0 <= total_machine_cpu_percent <= 100",
        "mean_total_machine_cpu_percent": sum(values) / len(values),
        "peak_total_machine_cpu_percent": max(values),
        "cpu_threads": payload["cpu_threads"],
        "logical_processors": payload["logical_processors"],
        "excluded_samples": [row for row in samples if row not in valid],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
