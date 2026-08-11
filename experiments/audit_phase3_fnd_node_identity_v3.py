"""Run the FND identity/role-energy audit with the Step 3 v3 policy adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.audit_phase3_fnd_node_identity as audit
import experiments.run_phase3_step3_v3 as adapter
from agents.ch_depletion_risk import validate_ch_risk_config
from baselines.policies import HTAMACPolicy


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ch-risk-config", type=Path, required=True)
    known, remaining = parser.parse_known_args()
    risk_path = known.ch_risk_config if known.ch_risk_config.is_absolute() else ROOT / known.ch_risk_config
    adapter._RISK = validate_ch_risk_config(json.loads(risk_path.read_text()))
    HTAMACPolicy.select_action = adapter.select_action_v3
    sys.argv = [sys.argv[0], *remaining]
    return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
