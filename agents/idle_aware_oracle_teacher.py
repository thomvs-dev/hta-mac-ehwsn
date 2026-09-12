"""Deterministic idle-aware repair of the frozen B4 HTA-MAC teacher.

The controller preserves the B4 packet count, stale service, projected Jain
fairness, minimum post-action member energy, caps, and active-member mask.  It
then exactly minimizes member transmit plus allocation-dependent idle-listening
energy inside the predeclared reduced neighborhood.  HEART cluster heads remain
exogenous.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agents.guarded_marginal_teacher import GuardedTeacherConfig, guarded_marginal_action
from experiments.run_v2_phase_b5_reduced_cluster_exact_oracle import exact_reduced_oracle


@dataclass(frozen=True)
class IdleAwareOracleTeacherConfig:
    base: GuardedTeacherConfig
    reduced_member_limit: int = 8

    @classmethod
    def from_payload(cls, payload: dict) -> "IdleAwareOracleTeacherConfig":
        limit = int(payload["reduced_member_limit"])
        if not 2 <= limit <= 8:
            raise ValueError("reduced_member_limit must lie in [2, 8]")
        return cls(
            base=GuardedTeacherConfig.from_payload(payload["base_teacher"]),
            reduced_member_limit=limit,
        )


def idle_aware_oracle_action(
    env,
    mask,
    caps,
    budget: int,
    config: IdleAwareOracleTeacherConfig,
    *,
    return_audit: bool = False,
):
    """Return the B4 action after an exact reduced-neighborhood energy repair."""
    base_action, base_audit = guarded_marginal_action(
        env, mask, caps, budget, config.base, return_audit=True
    )
    if int(base_action.sum()) == 0:
        audit = {
            "base_teacher": base_audit,
            "action_changed": False,
            "predicted_reduced_energy_saving_j": 0.0,
            "oracle": {"enumerated": 1, "feasible": 1, "local_members": []},
        }
        return (base_action, audit) if return_audit else base_action

    action, oracle_audit = exact_reduced_oracle(
        env, base_action, caps, mask, config.reduced_member_limit
    )
    saving = float(
        oracle_audit.get("base_reduced_tx_idle_j", 0.0)
        - oracle_audit.get("oracle_reduced_tx_idle_j", 0.0)
    )
    if saving < -1e-12:
        raise RuntimeError("idle-aware exact repair increased its frozen objective")
    audit = {
        "base_teacher": base_audit,
        "action_changed": bool(np.any(action != base_action)),
        "predicted_reduced_energy_saving_j": max(0.0, saving),
        "oracle": oracle_audit,
    }
    return (action, audit) if return_audit else action
