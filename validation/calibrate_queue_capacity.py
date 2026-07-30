"""Calibrate q_max from a static equal-slot pilot with TTL expiry."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ch_selection.frozen_heart_ch import FrozenHeartCH
from core.ch_selection.frozen_schedule_full import frozen_ch_schedule_full
from core.configuration import load_simple_yaml
from core.energy.radio_model import RadioModel
from core.hmm import load_solar_hmm, load_thermal_auxiliary
from envs import MACEnvironmentConfig
from envs.scheduled_mac_env import ScheduledIntraClusterMACEnv


def main() -> int:
    base = load_simple_yaml(ROOT / "config" / "base.yaml")
    config = load_simple_yaml(ROOT / "config" / "phase1.yaml")
    manifest = load_simple_yaml(ROOT / "core" / "frozen_assets.yaml")
    upstream = ROOT.parent / "final_repo"
    policy = FrozenHeartCH(
        upstream,
        manifest["checkpoint"]["path"],
        manifest["checkpoint"]["sha256"],
    )
    schedule_result = frozen_ch_schedule_full(policy, 2100, horizon=3000)
    bundle = dict(schedule_result["frames"][0])
    bundle["schedule"] = schedule_result["frames"]
    bundle["schedule_metadata"] = {
        key: value for key, value in schedule_result.items() if key != "frames"
    }
    solar = load_solar_hmm(upstream / manifest["solar_hmm"]["path"])
    thermal = load_thermal_auxiliary(
        ROOT / manifest["thermal_hmm"]["auxiliary_path"]
    )
    radio = RadioModel(
        e_elec_j_per_bit=base["radio"]["e_elec_j_per_bit"],
        eps_fs_j_per_bit_m2=base["radio"]["eps_fs_j_per_bit_m2"],
        eps_mp_j_per_bit_m4=base["radio"]["eps_mp_j_per_bit_m4"],
        e_da_j_per_bit=base["radio"]["e_da_j_per_bit"],
        d0_m=base["radio"]["d0_m"],
    )
    env_config = MACEnvironmentConfig(
        initial_energy_j=base["network"]["initial_energy_j"],
        packet_bits=base["network"]["packet_bits"],
        control_packet_bits=base["network"]["control_packet_bits"],
        e_elec_j_per_bit=base["radio"]["e_elec_j_per_bit"],
        frame_slot_budget=config["frame_slot_budget"],
        n_max=config["n_max"],
        queue_max_packets=config["queue_max_packets"],
        packet_ttl_rounds=config["packet_ttl_rounds"],
        max_rounds=config["max_rounds"],
        solar_scale=base["harvesting"]["solar"]["rectification_scale"],
        thermal_scale=base["harvesting"]["thermal"]["rectification_scale"],
        bs_position_m=tuple(base["network"]["bs_position_m"]),
        idle_slot_bit_times=config["idle_energy"]["primary_slot_bit_times"],
    )
    env = ScheduledIntraClusterMACEnv(
        env_config, radio, solar, thermal, idle_energy_enabled=False
    )
    env.reset(seed=2100, frozen_snapshot=bundle)
    censored = False
    while env.t_fnd is None and env.round < env.cfg.max_rounds:
        _, _, _, truncated, info = env.step(env.static_equal_action())
        if truncated and env.t_fnd is None:
            censored = True
            break
    observed = int(env.max_backlog_observed)
    recommended = observed + 1
    report = {
        "seed": 2100,
        "policy": "static_equal_round_robin",
        "idle_energy_enabled": False,
        "packet_ttl_rounds": env.cfg.packet_ttl_rounds,
        "calibration_queue_ceiling": env.cfg.queue_max_packets,
        "observed_max_backlog": observed,
        "recommended_queue_max_packets": recommended,
        "dropped_stale_packets": env.dropped_stale_packets,
        "dropped_overflow_packets": env.dropped_overflow_packets,
        "rounds_observed": env.round,
        "t_fnd": env.t_fnd,
        "right_censored_schedule": censored,
        "schedule_coverage_rounds": schedule_result["coverage_rounds"],
    }
    output = ROOT / "outputs" / "logs" / "queue_capacity_calibration.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"OBSERVED_MAX_BACKLOG={observed}")
    print(f"RECOMMENDED_Q_MAX={recommended}")
    print(f"STALE_DROPS={env.dropped_stale_packets}")
    print(f"report={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
