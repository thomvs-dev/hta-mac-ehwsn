"""Freeze the current HEART-CH thermal defaults as a provenance-labelled artifact."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--upstream", type=Path, default=PROJECT_ROOT.parent / "final_repo"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "core" / "hmm" / "thermal_auxiliary_params.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    upstream = args.upstream.resolve()
    sys.path.insert(0, str(upstream))
    import config as cfg

    states = int(cfg.NUM_THERMAL_STATES)
    transition = np.eye(states) * 0.8 + np.ones((states, states)) * (0.2 / states)
    initial = np.ones(states, dtype=np.float64) / states
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        thermal_A=transition,
        thermal_mu=np.asarray(cfg.DEFAULT_THERMAL_MU, dtype=np.float64),
        thermal_sigma2=np.asarray(cfg.DEFAULT_THERMAL_SIGMA2, dtype=np.float64),
        thermal_pi0=initial,
    )
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repository": str(upstream),
        "source": "config.py defaults and env/wsn_env.py transition construction",
        "provenance": "synthetic_auxiliary_from_heart_ch_defaults",
        "trained": False,
        "warning": "This artifact must not be described as a fitted thermal HMM.",
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(args.output)
    print(metadata_path)


if __name__ == "__main__":
    main()

