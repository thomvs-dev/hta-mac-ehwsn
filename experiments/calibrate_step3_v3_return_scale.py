"""Calibrate C51 support on the exact Step 3 v3 observation/reward/controller path."""

from __future__ import annotations

import argparse
import hashlib
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
from agents.qos_constraints_v3 import Step3QoSConstraintConfig, Step3QoSConstraintController
from envs.step3_lifetime_env import RoleSeparatedScheduledMACEnv, configure_step3_risk
from envs.step3_policy_observation import STEP3_CH_CONTEXT_SCHEMA
from envs.step3_v3_env import Step3V3DynamicClusterTrainingEnv


_RISK = None


class Step3V3RewardModel:
    def __init__(self, scales, weights):
        self.scales, self.weights = dict(scales), dict(weights)
        self.scales["ch_depletion_risk"] = float(_RISK["scale"])
        self.weights["ch_depletion_risk"] = float(_RISK["weight"])

    def evaluate(self, raw_terms):
        weighted = {
            name: self.weights[name] * float(raw_terms[name]) / max(self.scales[name], 1e-12)
            for name in trainer.TERM_ORDER
        }
        return float(sum(weighted.values())), weighted


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_v3_qos(path):
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = Step3QoSConstraintConfig.from_payload(payload)
    evidence_path = Path(payload["feasibility_evidence"])
    if not evidence_path.is_absolute():
        evidence_path = ROOT / evidence_path
    if not evidence_path.is_file():
        raise FileNotFoundError(evidence_path)
    return config, {
        "path": str(config_path.resolve()),
        "sha256": file_sha256(config_path),
        "feasibility_evidence_path": str(evidence_path.resolve()),
        "feasibility_evidence_sha256": file_sha256(evidence_path),
        "payload": payload,
    }


def main():
    global _RISK
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ch-risk-config", type=Path, required=True)
    known, remaining = parser.parse_known_args()
    risk_path = known.ch_risk_config if known.ch_risk_config.is_absolute() else ROOT / known.ch_risk_config
    _RISK = validate_ch_risk_config(json.loads(risk_path.read_text(encoding="utf-8")))
    configure_step3_risk(_RISK)
    trainer.ScheduledIntraClusterMACEnv = RoleSeparatedScheduledMACEnv
    trainer.DynamicClusterTrainingEnv = Step3V3DynamicClusterTrainingEnv
    trainer.TERM_ORDER = tuple(reward_module.TERM_ORDER) + ("ch_depletion_risk",)
    calibrator.PHASE2D_POLICY_SCHEMA = STEP3_CH_CONTEXT_SCHEMA
    calibrator.build_curriculum = trainer.build_curriculum
    calibrator.RewardModel = Step3V3RewardModel
    calibrator.QoSConstraintController = Step3QoSConstraintController
    calibrator.load_qos_constraints = load_v3_qos
    sys.argv = [sys.argv[0], *remaining]
    calibrator.main()


if __name__ == "__main__":
    main()
