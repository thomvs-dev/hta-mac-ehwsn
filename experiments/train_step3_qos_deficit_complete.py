"""Step 3 complete trainer with the frozen QoS-deficit action controller."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.train_step3_v3_complete as complete
from experiments.sweep_step3_qos_deficit_override import qos_deficit_override


_ACTIVE_ENV = None
_CONTROLLER_CONFIG = None
_CONTROLLER_TOTALS = {
    "triggered_steps": 0, "cumulative_deficit_after_base": 0,
    "requested_additions": 0, "available_opportunity": 0,
    "added_slots": 0, "risk_blocked_slots": 0,
}


class QoSDeficitTrainingEnv(complete.EnergyAuditedStep3V3Env):
    def reset(self):
        global _ACTIVE_ENV
        result = super().reset()
        _ACTIVE_ENV = self
        return result

    def step(self, action):
        global _ACTIVE_ENV
        _ACTIVE_ENV = self
        return super().step(action)


class QoSDeficitAgent(complete.v3.RecoveryExportAgent):
    def act(self, state, mask, epsilon=0.0, caps=None, budget=None, tie_break_priorities=None):
        action, q_values = super().act(
            state, mask, epsilon=epsilon, caps=caps, budget=budget,
            tie_break_priorities=tie_break_priorities,
        )
        env = _ACTIVE_ENV
        if env is None:
            raise RuntimeError("QoS-deficit controller has no active environment")
        completed, audit = qos_deficit_override(
            action, q_values, caps, mask, env,
            trajectory_target=_CONTROLLER_CONFIG["delivery_trajectory_target"],
            reserve_floor=_CONTROLLER_CONFIG["ch_post_forwarding_reserve_floor"],
            completion_fraction=_CONTROLLER_CONFIG["deficit_completion_fraction"],
        )
        _CONTROLLER_TOTALS["triggered_steps"] += audit["triggered"]
        _CONTROLLER_TOTALS["cumulative_deficit_after_base"] += audit["deficit_after_base"]
        _CONTROLLER_TOTALS["requested_additions"] += audit["requested_additions"]
        _CONTROLLER_TOTALS["available_opportunity"] += audit["opportunity"]
        _CONTROLLER_TOTALS["added_slots"] += audit["added"]
        _CONTROLLER_TOTALS["risk_blocked_slots"] += audit["risk_blocked"]
        return completed, q_values


def main():
    global _CONTROLLER_CONFIG
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--qos-deficit-config", type=Path, required=True)
    known, remaining = parser.parse_known_args()
    config_path = known.qos_deficit_config if known.qos_deficit_config.is_absolute() else ROOT / known.qos_deficit_config
    config = json.loads(config_path.read_text())
    if config.get("status") != "frozen_selected_qos_deficit_candidate":
        raise RuntimeError("QoS-deficit candidate is not frozen")
    sweep_path = ROOT / config["parent_sweep"]
    if complete.v3.sha256(sweep_path) != config["parent_sweep_sha256"]:
        raise RuntimeError("QoS-deficit parent sweep hash mismatch")
    sweep = json.loads(sweep_path.read_text())
    if sweep.get("selected_candidate") != config["candidate_id"]:
        raise RuntimeError("QoS-deficit candidate does not match frozen selection")
    for field in ("delivery_trajectory_target", "ch_post_forwarding_reserve_floor", "deficit_completion_fraction"):
        selected = next(row for row in sweep["candidates"] if row["candidate_id"] == config["candidate_id"])
        source_field = "reserve_floor" if field == "ch_post_forwarding_reserve_floor" else ("completion_fraction" if field == "deficit_completion_fraction" else field)
        if float(config[field]) != float(selected[source_field]):
            raise RuntimeError(f"QoS-deficit selection mismatch: {field}")
    if int(config["bounded_episodes"]) != 100 or int(config["optimizer_seed"]) != 5599:
        raise RuntimeError("bounded run contract changed")
    required_cli = {"--episodes": 100, "--max-steps": 1200, "--optimizer-seed": 5599}
    for flag, expected in required_cli.items():
        if flag not in remaining or int(remaining[remaining.index(flag) + 1]) != expected:
            raise RuntimeError(f"bounded CLI contract changed: {flag}")
    _CONTROLLER_CONFIG = config
    complete.EnergyAuditedStep3V3Env = QoSDeficitTrainingEnv
    complete.v3.RecoveryExportAgent = QoSDeficitAgent
    sys.argv = [sys.argv[0], *remaining]
    result = complete.main()
    run_name = remaining[remaining.index("--run-name") + 1]
    summary_path = ROOT / "outputs" / "phase2" / run_name / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        summary["qos_deficit_override"] = {
            "config_path": str(config_path.resolve()),
            "config_sha256": complete.v3.sha256(config_path),
            "candidate_id": config["candidate_id"],
            "totals": _CONTROLLER_TOTALS,
            "applied_during_training_and_greedy_evaluation": True,
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
