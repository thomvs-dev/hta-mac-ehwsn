"""Small pre-training T sweep around measured cluster contention."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

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
    common = MACEnvironmentConfig(
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
        idle_slot_bit_times=base["network"]["control_packet_bits"],
    )
    seeds = [2100, 2101, 2102]
    schedules = {}
    for seed in seeds:
        result = frozen_ch_schedule_full(policy, seed, horizon=3000)
        bundle = dict(result["frames"][0])
        bundle["schedule"] = result["frames"]
        bundle["schedule_metadata"] = {
            key: value for key, value in result.items() if key != "frames"
        }
        schedules[seed] = bundle

    candidates = [18, 22, 24, 27, 30]
    rows = []
    for budget in candidates:
        for seed in seeds:
            env = ScheduledIntraClusterMACEnv(
                replace(common, frame_slot_budget=budget),
                radio,
                solar,
                thermal,
                idle_energy_enabled=True,
            )
            env.reset(seed=seed, frozen_snapshot=schedules[seed])
            censored = False
            while env.t_fnd is None and env.round < env.cfg.max_rounds:
                _, _, _, truncated, _ = env.step(env.static_equal_action())
                if truncated and env.t_fnd is None:
                    censored = True
                    break
            rows.append(
                {
                    "T": budget,
                    "seed": seed,
                    "t_fnd": env.t_fnd,
                    "rounds": env.round,
                    "packets_delivered": env.total_packets,
                    "dropped_stale_packets": env.dropped_stale_packets,
                    "max_backlog": env.max_backlog_observed,
                    "right_censored": censored,
                }
            )
    summary = []
    for budget in candidates:
        selected = [row for row in rows if row["T"] == budget]
        summary.append(
            {
                "T": budget,
                "median_t_fnd": float(
                    np.median([row["t_fnd"] for row in selected])
                ),
                "median_packets_delivered": float(
                    np.median([row["packets_delivered"] for row in selected])
                ),
                "median_stale_drops": float(
                    np.median(
                        [row["dropped_stale_packets"] for row in selected]
                    )
                ),
                "censored_trials": int(
                    sum(row["right_censored"] for row in selected)
                ),
            }
        )
    report = {
        "status": "pilot_not_final_ablation",
        "idle_variant": "100_bit_control_header",
        "seeds": seeds,
        "primary_T_rule_selected": config["frame_slot_budget"],
        "candidates": candidates,
        "summary": summary,
        "trials": rows,
    }
    output = ROOT / "outputs" / "logs" / "pilot_t_sweep.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for row in summary:
        print(
            f"T={row['T']} MEDIAN_FND={row['median_t_fnd']:.1f} "
            f"MEDIAN_PACKETS={row['median_packets_delivered']:.1f}"
        )
    print(f"report={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
