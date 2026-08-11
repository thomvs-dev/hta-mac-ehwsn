"""Run the strict foundation audit against the Step 3 v3 observation schema."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.audit_phase2d_foundation as audit
import experiments.train_phase2_dynamic_curriculum as trainer
from envs.step3_lifetime_env import RoleSeparatedScheduledMACEnv, configure_step3_risk
from envs.step3_policy_observation import STEP3_CH_CONTEXT_SCHEMA
from envs.step3_v3_env import Step3V3DynamicClusterTrainingEnv


def main():
    configure_step3_risk(
        json.loads((ROOT / "config" / "step3_v3_risk_weight_1.json").read_text())
    )
    trainer.ScheduledIntraClusterMACEnv = RoleSeparatedScheduledMACEnv
    trainer.DynamicClusterTrainingEnv = Step3V3DynamicClusterTrainingEnv
    trainer.PHASE2D_POLICY_SCHEMA = STEP3_CH_CONTEXT_SCHEMA
    audit.PHASE2D_POLICY_SCHEMA = STEP3_CH_CONTEXT_SCHEMA
    audit.build_curriculum = trainer.build_curriculum
    return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
