"""Authoritative post-instructor Phase 1 inspection gate."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import validation.phase1_gate_pre_revision as gate
from core.ch_selection.frozen_schedule_full import frozen_ch_schedule_full
from envs.scheduled_mac_env import ScheduledIntraClusterMACEnv


SCHEDULE_CACHE: dict[int, dict] = {}


def _schedule_bundle(policy, seed):
    result = frozen_ch_schedule_full(policy, seed, horizon=3000)
    SCHEDULE_CACHE[int(seed)] = result
    bundle = dict(result["frames"][0])
    bundle["schedule"] = result["frames"]
    bundle["schedule_metadata"] = {
        key: value for key, value in result.items() if key != "frames"
    }
    return bundle


def _run_to_fnd(env, bundle, seed):
    env.reset(seed=seed, frozen_snapshot=bundle)
    censored = False
    while env.t_fnd is None and env.round < env.cfg.max_rounds:
        _, _, _, truncated, _ = env.step(env.static_equal_action())
        if truncated and env.t_fnd is None:
            censored = True
            break
    return env.t_fnd, censored


def main() -> int:
    gate.frozen_initial_snapshot = _schedule_bundle
    gate.IntraClusterMACEnv = ScheduledIntraClusterMACEnv
    legacy_exit = gate.main()

    cfg, radio, solar, thermal, _ = gate._make_components()
    control_cfg = replace(
        cfg,
        idle_slot_bit_times=cfg.control_packet_bits,
    )
    control_fnd = []
    control_censored = 0
    for seed in range(2100, 2105):
        result = SCHEDULE_CACHE[seed]
        bundle = dict(result["frames"][0])
        bundle["schedule"] = result["frames"]
        bundle["schedule_metadata"] = {
            key: value for key, value in result.items() if key != "frames"
        }
        env = ScheduledIntraClusterMACEnv(
            control_cfg,
            radio,
            solar,
            thermal,
            idle_energy_enabled=True,
        )
        fnd, censored = _run_to_fnd(env, bundle, seed)
        control_censored += int(censored)
        control_fnd.append(
            control_cfg.max_rounds if fnd is None else int(fnd)
        )

    report_path = ROOT / "outputs" / "logs" / "phase1_gate.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    primary = report["idle_ablation"]["median_network_t_fnd_on"]
    idle_off = report["idle_ablation"]["median_network_t_fnd_off"]
    control_median = float(np.median(control_fnd))
    sensitivity_pass = primary < control_median < idle_off
    report["authoritative_post_instructor_revision"] = True
    report["state_semantics"] = {
        "probability_blocks": "state-conditioned transition probabilities",
        "bayesian_posterior_claim": False,
        "forecast_moments": "HEART-CH manuscript Eqs. 13-14 rectified Gaussian",
        "feature_validation": "outputs/logs/harvest_feature_validation.json",
    }
    report["queue_model"] = {
        "one_packet_generated_per_alive_node_round": True,
        "packet_ttl_rounds": cfg.packet_ttl_rounds,
        "queue_max_packets": cfg.queue_max_packets,
        "calibration": "outputs/logs/queue_capacity_calibration.json",
    }
    report["idle_sensitivity"] = {
        "primary_full_data_slot_bit_times": cfg.idle_slot_bit_times,
        "secondary_control_header_bit_times": cfg.control_packet_bits,
        "idle_off_bit_times": 0,
        "primary_median_t_fnd": primary,
        "control_header_median_t_fnd": control_median,
        "idle_off_median_t_fnd": idle_off,
        "control_header_trial_t_fnd": control_fnd,
        "control_header_censored_trials": control_censored,
        "expected_order_pass": sensitivity_pass,
    }
    report["schedule_coverage"] = [
        {
            "seed": seed,
            "requested_horizon": result["requested_horizon"],
            "coverage_rounds": result["coverage_rounds"],
            "complete": result["complete"],
            "stop_reason": result["stop_reason"],
            "stale_frame_replay": False,
        }
        for seed, result in sorted(SCHEDULE_CACHE.items())
    ]
    report["cluster_integration"] = {
        "mode": "shared per-round frozen HEART-CH schedule replay",
        "shared_paired_schedule": True,
        "embedding_mode": "shared exogenous schedule replay",
        "ch_retraining": False,
        "routing_changes": False,
        "schedule_exhaustion": "right censor; never repeat last frame",
    }
    if not sensitivity_pass:
        report["status"] = "fail"
        report.setdefault("failures", []).append(
            "idle sensitivity does not satisfy data-slot < header-only < off"
        )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        "IDLE_SENSITIVITY_MEDIANS="
        f"{primary:.1f}/{control_median:.1f}/{idle_off:.1f}"
    )
    print(f"IDLE_SENSITIVITY_ORDER_PASS={sensitivity_pass}")
    print(f"AUTHORITATIVE_PHASE_1_GATE={report['status'].upper()}")
    return 0 if legacy_exit == 0 and sensitivity_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
