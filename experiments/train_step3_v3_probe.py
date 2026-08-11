"""Create a bounded Step 3 v3 checkpoint for same-platform foundation audit."""

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
from agents.qos_constraints_v3 import Step3QoSConstraintConfig, Step3QoSConstraintController
from core.runtime_fingerprint import validate_runtime_contract
from envs.step3_lifetime_env import RoleSeparatedScheduledMACEnv, configure_step3_risk
from envs.step3_policy_observation import STEP3_CH_CONTEXT_SCHEMA
from envs.step3_v3_env import Step3V3DynamicClusterTrainingEnv

_RISK = None


class Reward:
    def __init__(self, scales, weights):
        self.scales, self.weights = dict(scales), dict(weights)
        self.scales["ch_depletion_risk"] = float(_RISK["scale"])
        self.weights["ch_depletion_risk"] = float(_RISK["weight"])

    def evaluate(self, raw):
        weighted = {name: self.weights[name] * float(raw[name]) / max(self.scales[name], 1e-12) for name in trainer.TERM_ORDER}
        return float(sum(weighted.values())), weighted


def main():
    global _RISK
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ch-risk-config", type=Path, required=True)
    parser.add_argument("--step3-qos-config", type=Path, required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    known, remaining = parser.parse_known_args()
    resolve = lambda p: p if p.is_absolute() else ROOT / p
    risk, qos, contract = map(resolve, (known.ch_risk_config, known.step3_qos_config, known.runtime_contract))
    _RISK = validate_ch_risk_config(json.loads(risk.read_text()))
    qos_payload = json.loads(qos.read_text())
    Step3QoSConstraintConfig.from_payload(qos_payload)
    runtime = validate_runtime_contract(json.loads(contract.read_text()))
    if not runtime["pass"]:
        raise RuntimeError(f"runtime mismatch: {runtime}")
    episodes = int(remaining[remaining.index("--episodes") + 1])
    max_steps = int(remaining[remaining.index("--max-steps") + 1])
    if not (1 <= episodes <= 5 and 1 <= max_steps <= 10):
        raise ValueError("v3 foundation probe is limited to 5 episodes and 10 steps")
    configure_step3_risk(_RISK)
    trainer.ScheduledIntraClusterMACEnv = RoleSeparatedScheduledMACEnv
    trainer.DynamicClusterTrainingEnv = Step3V3DynamicClusterTrainingEnv
    trainer.PHASE2D_POLICY_SCHEMA = STEP3_CH_CONTEXT_SCHEMA
    trainer.TERM_ORDER = tuple(reward_module.TERM_ORDER) + ("ch_depletion_risk",)
    trainer.RewardModel = Reward
    trainer.QoSConstraintConfig = Step3QoSConstraintConfig
    trainer.QoSConstraintController = Step3QoSConstraintController
    if "--qos-constraint-config" not in remaining:
        remaining.extend(["--qos-constraint-config", str(qos)])
    sys.argv = [sys.argv[0], *remaining]
    return trainer.main()


if __name__ == "__main__":
    raise SystemExit(main())
