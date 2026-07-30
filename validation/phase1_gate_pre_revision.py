"""Executable Phase 1 inspection gate."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.budget_projection import project_slot_budget
from core.ch_selection.frozen_heart_ch import FrozenHeartCH
from core.ch_selection.initial_snapshot import frozen_initial_snapshot
from core.configuration import load_simple_yaml
from core.energy.radio_model import RadioModel
from core.hmm import load_solar_hmm, load_thermal_auxiliary
from envs import IntraClusterMACEnv, MACEnvironmentConfig


def _make_components():
    base = load_simple_yaml(ROOT / "config" / "base.yaml")
    phase1 = load_simple_yaml(ROOT / "config" / "phase1.yaml")
    manifest = load_simple_yaml(ROOT / "core" / "frozen_assets.yaml")
    upstream = ROOT.parent / "final_repo"
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
        e_elec_j_per_bit=base["radio"]["e_elec_j_per_bit"],
        frame_slot_budget=phase1["frame_slot_budget"],
        n_max=phase1["n_max"],
        queue_max_packets=phase1["queue_max_packets"],
        max_rounds=phase1["max_rounds"],
        solar_scale=base["harvesting"]["solar"]["rectification_scale"],
        thermal_scale=base["harvesting"]["thermal"]["rectification_scale"],
        bs_position_m=tuple(base["network"]["bs_position_m"]),
    )
    frozen = FrozenHeartCH(
        upstream,
        manifest["checkpoint"]["path"],
        manifest["checkpoint"]["sha256"],
    )
    return cfg, radio, solar, thermal, frozen


def _run_to_fnd(env, snapshot, seed):
    env.reset(seed=seed, frozen_snapshot=snapshot)
    member_fnd = None
    initial_ch = set(int(v) for v in snapshot["cluster_heads"])
    while env.t_fnd is None and env.round < env.cfg.max_rounds:
        env.step(env.static_equal_action())
        if member_fnd is None:
            dead_members = [
                i for i, alive in enumerate(env.alive)
                if not alive and i not in initial_ch
            ]
            if dead_members:
                member_fnd = env.round
    return env.t_fnd or env.cfg.max_rounds, member_fnd or env.cfg.max_rounds


def _emission_samples(params, scale, seed, count):
    rng = np.random.default_rng(seed)
    states = rng.choice(len(params.mean), size=count, p=params.initial)
    raw = rng.normal(params.mean[states], np.sqrt(params.variance[states]))
    return np.maximum(0.0, raw * scale)


def main() -> int:
    cfg, radio, solar, thermal, frozen = _make_components()
    failures: list[str] = []

    snapshot = frozen_initial_snapshot(frozen, seed=2026)
    if len(snapshot["cluster_heads"]) != 5:
        failures.append("frozen checkpoint did not select K=5")

    env = IntraClusterMACEnv(cfg, radio, solar, thermal)
    state, _ = env.reset(seed=2026, frozen_snapshot=snapshot)
    if state.shape != (100, 18):
        failures.append(f"state shape is {state.shape}, expected (100,18)")
    if not np.allclose(state[:, 3:11].sum(axis=1), 1.0):
        failures.append("solar posterior block does not sum to one")
    if not np.allclose(state[:, 11:15].sum(axis=1), 1.0):
        failures.append("thermal posterior block does not sum to one")

    max_energy_error = 0.0
    traces = []
    for _ in range(20):
        _, _, _, _, info = env.step(env.static_equal_action())
        trace = info["energy_trace"]
        expected = np.clip(
            trace["energy_before"] - trace["consumed"] + trace["harvested"],
            0.0,
            cfg.initial_energy_j,
        )
        error = float(np.max(np.abs(expected - trace["energy_after"])))
        max_energy_error = max(max_energy_error, error)
        traces.append(
            {
                "round": trace["round"],
                "consumed_j": float(trace["consumed"].sum()),
                "harvested_j": float(trace["harvested"].sum()),
                "energy_after_j": float(trace["energy_after"].sum()),
                "max_abs_error_j": error,
            }
        )
    if max_energy_error > 1e-12:
        failures.append(f"20-round energy trace error {max_energy_error}")

    env_a = IntraClusterMACEnv(cfg, radio, solar, thermal)
    env_b = IntraClusterMACEnv(cfg, radio, solar, thermal)
    state_a, _ = env_a.reset(seed=777, frozen_snapshot=snapshot)
    state_b, _ = env_b.reset(seed=777, frozen_snapshot=snapshot)
    deterministic = np.array_equal(state_a, state_b)
    for _ in range(20):
        out_a = env_a.step(env_a.static_equal_action())
        out_b = env_b.step(env_b.static_equal_action())
        deterministic &= np.array_equal(out_a[0], out_b[0])
        deterministic &= np.array_equal(env_a.energy, env_b.energy)
    if not deterministic:
        failures.append("same-seed trajectories differ")

    rng = np.random.default_rng(991)
    budget_violations = 0
    for _ in range(1000):
        nodes = int(rng.integers(1, 31))
        q_values = rng.normal(size=(nodes, cfg.n_max + 1))
        allocation = project_slot_budget(q_values, cfg.frame_slot_budget)
        if (
            int(allocation.sum()) > cfg.frame_slot_budget
            or np.any(allocation < 0)
            or np.any(allocation > cfg.n_max)
        ):
            budget_violations += 1
    if budget_violations:
        failures.append(f"{budget_violations} budget violations")

    solar_new = _emission_samples(solar, cfg.solar_scale, 100, 10_000)
    solar_reference = _emission_samples(solar, cfg.solar_scale, 200, 10_000)
    thermal_new = _emission_samples(thermal, cfg.thermal_scale, 300, 10_000)
    thermal_reference = _emission_samples(
        thermal, cfg.thermal_scale, 400, 10_000
    )
    solar_ks = ks_2samp(solar_new, solar_reference)
    thermal_ks = ks_2samp(thermal_new, thermal_reference)
    if solar_ks.pvalue <= 0.05:
        failures.append(f"solar KS p={solar_ks.pvalue}")
    if thermal_ks.pvalue <= 0.05:
        failures.append(f"thermal KS p={thermal_ks.pvalue}")

    seeds = list(range(2100, 2105))
    network_on, network_off, member_on, member_off = [], [], [], []
    trial_rows = []
    for seed in seeds:
        snap = frozen_initial_snapshot(frozen, seed=seed)
        on = IntraClusterMACEnv(
            cfg, radio, solar, thermal, idle_energy_enabled=True
        )
        off = IntraClusterMACEnv(
            cfg, radio, solar, thermal, idle_energy_enabled=False
        )
        fnd_on, mfnd_on = _run_to_fnd(on, snap, seed)
        fnd_off, mfnd_off = _run_to_fnd(off, snap, seed)
        network_on.append(fnd_on)
        network_off.append(fnd_off)
        member_on.append(mfnd_on)
        member_off.append(mfnd_off)
        trial_rows.append(
            {
                "seed": seed,
                "t_fnd_idle_on": fnd_on,
                "t_fnd_idle_off": fnd_off,
                "member_t_fnd_idle_on": mfnd_on,
                "member_t_fnd_idle_off": mfnd_off,
                "idle_energy_on_j": on.total_idle_energy,
                "idle_energy_off_j": off.total_idle_energy,
            }
        )
    network_shift = float(np.median(network_off) - np.median(network_on))
    member_shift = float(np.median(member_off) - np.median(member_on))
    if network_shift <= 0:
        failures.append(
            "idle on/off produced no positive network T_FND shift"
        )
    if not all(row["idle_energy_on_j"] > 0 for row in trial_rows):
        failures.append("idle-on run accumulated no idle energy")
    if not all(row["idle_energy_off_j"] == 0 for row in trial_rows):
        failures.append("idle-off run accumulated idle energy")

    report = {
        "phase": 1,
        "status": "pass" if not failures else "fail",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_ch_count": int(len(snapshot["cluster_heads"])),
        "state_shape": list(state.shape),
        "energy_conservation_20_rounds": {
            "max_abs_error_j": max_energy_error,
            "trace": traces,
        },
        "determinism_same_seed_20_rounds": bool(deterministic),
        "budget_projection": {
            "random_cases": 1000,
            "violations": budget_violations,
        },
        "hmm_ks_10000": {
            "solar_statistic": float(solar_ks.statistic),
            "solar_pvalue": float(solar_ks.pvalue),
            "thermal_statistic": float(thermal_ks.statistic),
            "thermal_pvalue": float(thermal_ks.pvalue),
        },
        "idle_ablation": {
            "trials": trial_rows,
            "median_network_t_fnd_on": float(np.median(network_on)),
            "median_network_t_fnd_off": float(np.median(network_off)),
            "median_network_shift_rounds": network_shift,
            "median_member_t_fnd_on": float(np.median(member_on)),
            "median_member_t_fnd_off": float(np.median(member_off)),
            "median_member_shift_rounds": member_shift,
        },
        "failures": failures,
    }
    output = ROOT / "outputs" / "logs" / "phase1_gate.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"FROZEN_CH_COUNT={len(snapshot['cluster_heads'])}")
    print(f"ENERGY_TRACE_MAX_ERROR_J={max_energy_error:.3e}")
    print(f"DETERMINISTIC={deterministic}")
    print(f"BUDGET_VIOLATIONS={budget_violations}/1000")
    print(
        f"HMM_KS_SOLAR_D={solar_ks.statistic:.4f},P={solar_ks.pvalue:.6f}"
    )
    print(
        f"HMM_KS_THERMAL_D={thermal_ks.statistic:.4f},P={thermal_ks.pvalue:.6f}"
    )
    print(
        "T_FND_IDLE_ON_OFF_MEDIAN="
        f"{np.median(network_on):.1f}/{np.median(network_off):.1f},"
        f"SHIFT={network_shift:.1f}"
    )
    print(f"MEMBER_T_FND_SHIFT={member_shift:.1f}")
    print(f"report={output}")
    print(f"PHASE_1_GATE={report['status'].upper()}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
