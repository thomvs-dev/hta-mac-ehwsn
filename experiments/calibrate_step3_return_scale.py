"""Calibrate Step 3 C51 support using only frozen development environments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agents.reward_model as reward_module
import experiments.calibrate_paper_aligned_return_scale as calibrator
import experiments.train_phase2_dynamic_curriculum as trainer
from agents.ch_depletion_risk import validate_ch_risk_config
from envs.step3_lifetime_env import (
    RoleSeparatedScheduledMACEnv,
    Step3DynamicClusterTrainingEnv,
    configure_step3_risk,
)


class Step3RewardModel:
    def __init__(self, scales, weights):
        config = _RISK_CONFIG
        self.scales = dict(scales)
        self.weights = dict(weights)
        self.scales["ch_depletion_risk"] = float(config["scale"])
        self.weights["ch_depletion_risk"] = float(config["weight"])

    def evaluate(self, raw_terms):
        weighted = {
            name: self.weights[name] * float(raw_terms[name]) / max(self.scales[name], 1e-12)
            for name in trainer.TERM_ORDER
        }
        return float(sum(weighted.values())), weighted


_RISK_CONFIG = None


def main() -> None:
    global _RISK_CONFIG
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ch-risk-config", required=True)
    known, remaining = parser.parse_known_args()
    risk_path = Path(known.ch_risk_config)
    if not risk_path.is_absolute():
        risk_path = ROOT / risk_path
    _RISK_CONFIG = validate_ch_risk_config(
        json.loads(risk_path.read_text(encoding="utf-8"))
    )
    configure_step3_risk(_RISK_CONFIG)
    trainer.ScheduledIntraClusterMACEnv = RoleSeparatedScheduledMACEnv
    trainer.DynamicClusterTrainingEnv = Step3DynamicClusterTrainingEnv
    trainer.TERM_ORDER = tuple(reward_module.TERM_ORDER) + ("ch_depletion_risk",)
    calibrator.build_curriculum = trainer.build_curriculum
    calibrator.RewardModel = Step3RewardModel
    sys.argv = [sys.argv[0], *remaining]
    calibrator.main()


if __name__ == "__main__":
    main()
