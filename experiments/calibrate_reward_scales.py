"""Measure Phase 2 reward scales before assigning weights."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.budget_projection import project_slot_budget
from agents.reward_model import TERM_ORDER
from core.ch_selection.frozen_heart_ch import FrozenHeartCH
from core.ch_selection.initial_snapshot import frozen_initial_snapshot
from core.configuration import load_simple_yaml
from core.energy.radio_model import RadioModel
from core.hmm import load_solar_hmm, load_thermal_auxiliary
from envs import IntraClusterMACEnv, MACEnvironmentConfig
from envs.fixed_cluster_training_env import FixedClusterTrainingEnv


def _setup():
    base = load_simple_yaml(ROOT / "config" / "base.yaml")
    mac = load_simple_yaml(ROOT / "config" / "phase1.yaml")
    manifest = load_simple_yaml(ROOT / "core" / "frozen_assets.yaml")
    upstream = ROOT.parent / "final_repo"
    policy = FrozenHeartCH(
        upstream,
        manifest["checkpoint"]["path"],
        manifest["checkpoint"]["sha256"],
    )
    snapshot = frozen_initial_snapshot(policy, 3210)
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
    cfg = MACEnvironmentConfig(
        initial_energy_j=base["network"]["initial_energy_j"],
        packet_bits=base["network"]["packet_bits"],
        control_packet_bits=base["network"]["control_packet_bits"],
        e_elec_j_per_bit=base["radio"]["e_elec_j_per_bit"],
        frame_slot_budget=mac["frame_slot_budget"],
        n_max=mac["n_max"],
        queue_max_packets=mac["queue_max_packets"],
        packet_ttl_rounds=mac["packet_ttl_rounds"],
        max_rounds=150,
        solar_scale=base["harvesting"]["solar"]["rectification_scale"],
        thermal_scale=base["harvesting"]["thermal"]["rectification_scale"],
        bs_position_m=tuple(base["network"]["bs_position_m"]),
        idle_slot_bit_times=mac["idle_energy"]["primary_slot_bit_times"],
    )
    return snapshot, solar, thermal, radio, cfg


def main() -> int:
    snapshot, solar, thermal, radio, cfg = _setup()
    rng = np.random.default_rng(8821)
    samples = {name: [] for name in TERM_ORDER}
    policy_rows = []
    for policy_name in ("always_sleep", "static_equal", "random_budgeted"):
        base_env = IntraClusterMACEnv(cfg, radio, solar, thermal)
        env = FixedClusterTrainingEnv(base_env, snapshot, seed=3210)
        _, mask, _ = env.reset()
        steps = 0
        while steps < 100:
            if policy_name == "always_sleep":
                action = np.zeros(env.member_count, dtype=np.int64)
            elif policy_name == "static_equal":
                action = np.zeros(env.member_count, dtype=np.int64)
                action[np.flatnonzero(mask)[: cfg.frame_slot_budget]] = 1
            else:
                q = rng.normal(size=(env.member_count, cfg.n_max + 1))
                q[:, 1:] += 1.0
                action = project_slot_budget(
                    q, cfg.frame_slot_budget, stop_at_nonpositive_gain=False
                )
                action[~mask] = 0
            _, mask, done, info = env.step(action)
            for name, value in info["reward_raw_terms"].items():
                samples[name].append(float(value))
            steps += 1
            if done:
                break
        policy_rows.append(
            {
                "policy": policy_name,
                "steps": steps,
                "packets": base_env.total_packets,
                "t_fnd": base_env.t_fnd,
                "target_member_count": env.member_count,
            }
        )

    scales = {}
    diagnostics = {}
    for name in TERM_ORDER:
        values = np.abs(np.asarray(samples[name], dtype=np.float64))
        nonzero = values[values > 0.0]
        scale = float(np.percentile(nonzero, 90)) if nonzero.size else 1.0
        scales[name] = max(scale, 1e-6)
        diagnostics[name] = {
            "min": float(np.min(samples[name])),
            "median": float(np.median(samples[name])),
            "max": float(np.max(samples[name])),
            "p90_abs_nonzero": scales[name],
        }

    base_importance = {
        "packets_delivered": 2.0,
        "idle_energy_j": 1.0,
        "deaths": 2.0,
        "high_harvest_alignment": 0.5,
        "declining_allocation": 0.5,
        "queue_fairness": 0.5,
    }
    report = {
        "status": "calibrated_not_trained",
        "seed": 3210,
        "policies": policy_rows,
        "raw_term_diagnostics": diagnostics,
        "scales": scales,
        "weights_after_scale_normalization": base_importance,
        "reward_equation": "sum(weight[name] * raw[name] / scale[name])",
    }
    output = ROOT / "outputs" / "logs" / "phase2_reward_calibration.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    config_output = ROOT / "config" / "reward_calibration.json"
    config_output.write_text(
        json.dumps(
            {
                "scales": scales,
                "weights": base_importance,
                "evidence": str(output.relative_to(ROOT)).replace("\\", "/"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"TARGET_CLUSTER_MEMBERS={policy_rows[0]['target_member_count']}")
    for name in TERM_ORDER:
        print(f"{name.upper()}_SCALE={scales[name]:.8g}")
    print(f"report={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
