"""Bounded Step 3 probe permitted before (and only before) full preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agents.reward_model as reward_module
import experiments.train_phase2_dynamic_curriculum as trainer
from agents.ch_depletion_risk import validate_ch_risk_config
from core.runtime_fingerprint import validate_runtime_contract
from envs.step3_lifetime_env import (
    RoleSeparatedScheduledMACEnv,
    Step3DynamicClusterTrainingEnv,
    configure_step3_risk,
)


_RISK_CONFIG = None


class Step3ProbeRewardModel:
    def __init__(self, scales, weights):
        self.scales = dict(scales)
        self.weights = dict(weights)
        self.scales["ch_depletion_risk"] = float(_RISK_CONFIG["scale"])
        self.weights["ch_depletion_risk"] = float(_RISK_CONFIG["weight"])

    def evaluate(self, raw_terms):
        weighted = {
            name: self.weights[name] * float(raw_terms[name]) / max(self.scales[name], 1e-12)
            for name in trainer.TERM_ORDER
        }
        return float(sum(weighted.values())), weighted


def main() -> int:
    global _RISK_CONFIG
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ch-risk-config", type=Path, required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    known, remaining = parser.parse_known_args()
    if "--episodes" not in remaining:
        raise ValueError("bounded probe requires explicit --episodes")
    episodes = int(remaining[remaining.index("--episodes") + 1])
    if not 1 <= episodes <= 5:
        raise ValueError("preflight probe is hard-limited to 1-5 episodes")
    resolve = lambda p: p if p.is_absolute() else ROOT / p
    risk_path, contract_path = resolve(known.ch_risk_config), resolve(known.runtime_contract)
    _RISK_CONFIG = validate_ch_risk_config(json.loads(risk_path.read_text()))
    validation = validate_runtime_contract(json.loads(contract_path.read_text()))
    if not validation["pass"]:
        raise RuntimeError(f"probe runtime mismatch: {validation}")
    configure_step3_risk(_RISK_CONFIG)
    trainer.ScheduledIntraClusterMACEnv = RoleSeparatedScheduledMACEnv
    trainer.DynamicClusterTrainingEnv = Step3DynamicClusterTrainingEnv
    trainer.TERM_ORDER = tuple(reward_module.TERM_ORDER) + ("ch_depletion_risk",)
    trainer.RewardModel = Step3ProbeRewardModel
    sys.argv = [sys.argv[0], *remaining]
    return trainer.main()


if __name__ == "__main__":
    raise SystemExit(main())
