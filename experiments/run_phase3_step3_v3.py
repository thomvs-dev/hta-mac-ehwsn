"""Run paired Phase 3 evaluation with the Step 3 v3 policy observation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_phase3_pilot as pilot
from agents.ch_depletion_risk import validate_ch_risk_config
from baselines.policies import HTAMACPolicy
from envs.step3_policy_observation import STEP3_CH_CONTEXT_SCHEMA, build_step3_observation


_RISK = None
_ORIGINAL_SELECT = HTAMACPolicy.select_action


def select_action_v3(self, state, env):
    if self.agent.cfg.state_schema != STEP3_CH_CONTEXT_SCHEMA:
        return _ORIGINAL_SELECT(self, state, env)
    action = np.zeros(env.n_nodes, dtype=np.int64)
    for cluster, ch in enumerate(env.cluster_heads):
        members = self.eligible_members(env, cluster, int(ch))
        if not len(members):
            continue
        cluster_mask = np.zeros(env.n_nodes, dtype=bool)
        cluster_mask[members] = True
        features = build_step3_observation(
            env, state, cluster_mask, ch=int(ch), members=members,
            risk_config=_RISK,
        )
        caps = np.minimum(env.queue, env.cfg.n_max)
        caps[~cluster_mask] = 0
        budget = min(
            env.cfg.frame_slot_budget,
            self.agent.cfg.budget if self.allocation_budget is None else self.allocation_budget,
        )
        global_action, _ = self.agent.act(
            features, cluster_mask, epsilon=0.0, caps=caps, budget=budget,
            tie_break_priorities=np.arange(env.n_nodes, dtype=np.int64),
        )
        action[members] = global_action[members]
    return self.validate(action, env)


def main():
    global _RISK
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ch-risk-config", type=Path, required=True)
    known, remaining = parser.parse_known_args()
    risk_path = known.ch_risk_config if known.ch_risk_config.is_absolute() else ROOT / known.ch_risk_config
    _RISK = validate_ch_risk_config(json.loads(risk_path.read_text(encoding="utf-8")))
    HTAMACPolicy.select_action = select_action_v3
    sys.argv = [sys.argv[0], *remaining]
    return pilot.main()


if __name__ == "__main__":
    raise SystemExit(main())
