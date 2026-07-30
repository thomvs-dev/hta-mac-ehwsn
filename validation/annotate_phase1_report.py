"""Attach non-negotiable scope provenance to the Phase 1 gate report."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs" / "logs" / "phase1_gate.json"


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    report["idle_ablation"]["scope"] = (
        "diagnostic member-idle isolation; CH Rx and aggregation energy held "
        "exogenous; CH-to-BS Tx retained; not a protocol-performance result"
    )
    report["idle_ablation"]["full_accounting_status"] = (
        "confounded by fixed-CH first death; requires per-round frozen CH "
        "integration before comparative evaluation"
    )
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"annotated={REPORT}")


if __name__ == "__main__":
    main()
