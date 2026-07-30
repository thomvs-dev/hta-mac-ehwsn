"""Run the Phase 3 paired scheduled-policy pilot without inferential claims."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines import StaticEqualPolicy, phase3_policy_factories
from core.ch_selection.frozen_heart_ch import FrozenHeartCH
from core.ch_selection.frozen_schedule_full import frozen_ch_schedule_full
from core.configuration import load_simple_yaml
from core.energy.radio_model import RadioModel
from core.hmm import load_solar_hmm, load_thermal_auxiliary
from envs import MACEnvironmentConfig
from envs.scheduled_mac_env import ScheduledIntraClusterMACEnv


METRICS = (
    "t_fnd",
    "t_hnd",
    "rounds",
    "throughput",
    "idle_energy_j",
    "queue_fairness",
    "energy_consumed_j",
    "energy_efficiency_packets_per_j",
    "dropped_stale_packets",
    "dropped_overflow_packets",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="3100,3101,3102,3103,3104")
    parser.add_argument("--horizon", type=int, default=3000)
    parser.add_argument("--run-name", default="paired_pilot_5seed")
    parser.add_argument("--skip-compatibility", action="store_true")
    return parser.parse_args()


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def build_assets(horizon: int):
    base = load_simple_yaml(ROOT / "config" / "base.yaml")
    mac = load_simple_yaml(ROOT / "config" / "phase1.yaml")
    manifest = load_simple_yaml(ROOT / "core" / "frozen_assets.yaml")
    upstream = ROOT.parent / "final_repo"
    frozen_policy = FrozenHeartCH(
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
    config = MACEnvironmentConfig(
        initial_energy_j=base["network"]["initial_energy_j"],
        packet_bits=base["network"]["packet_bits"],
        control_packet_bits=base["network"]["control_packet_bits"],
        e_elec_j_per_bit=base["radio"]["e_elec_j_per_bit"],
        frame_slot_budget=mac["frame_slot_budget"],
        n_max=mac["n_max"],
        queue_max_packets=mac["queue_max_packets"],
        packet_ttl_rounds=mac["packet_ttl_rounds"],
        max_rounds=horizon,
        solar_scale=base["harvesting"]["solar"]["rectification_scale"],
        thermal_scale=base["harvesting"]["thermal"]["rectification_scale"],
        bs_position_m=tuple(base["network"]["bs_position_m"]),
        idle_slot_bit_times=mac["idle_energy"]["primary_slot_bit_times"],
    )
    return frozen_policy, solar, thermal, radio, config, manifest


def schedule_bundle(policy, seed: int, horizon: int, checkpoint_sha: str):
    cache_dir = ROOT / "outputs" / "cache" / "phase3_schedules"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"seed_{seed}_horizon_{horizon}_{checkpoint_sha[:12]}.pkl"
    if cache.exists():
        with cache.open("rb") as handle:
            result = pickle.load(handle)
        source = "cache"
    else:
        print(f"GENERATING_SCHEDULE seed={seed} horizon={horizon}", flush=True)
        result = frozen_ch_schedule_full(policy, seed, horizon=horizon)
        with cache.open("wb") as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        source = "generated"
    bundle = dict(result["frames"][0])
    bundle["schedule"] = result["frames"]
    bundle["schedule_metadata"] = {
        key: value for key, value in result.items() if key != "frames"
    }
    metadata = dict(bundle["schedule_metadata"])
    metadata["cache_source"] = source
    metadata["cache_file"] = str(cache.relative_to(ROOT)).replace("\\", "/")
    return bundle, metadata


def jain(values):
    values = np.asarray(values, dtype=np.float64)
    denominator = len(values) * np.square(values).sum()
    return float(values.sum() ** 2 / denominator) if denominator > 0.0 else 0.0


def run_one(policy, seed, bundle, solar, thermal, radio, config, idle_enabled):
    env = ScheduledIntraClusterMACEnv(
        config, radio, solar, thermal, idle_energy_enabled=idle_enabled
    )
    policy.reset(seed)
    state, _ = env.reset(seed=seed, frozen_snapshot=bundle)
    cumulative_service = np.zeros(env.n_nodes, dtype=np.float64)
    total_consumed = 0.0
    hnd = None
    terminated = False
    truncated = False
    info = env._info()
    while not terminated and not truncated:
        action = policy.select_action(state, env)
        state, _, terminated, truncated, info = env.step(action)
        delivered = np.asarray(
            info["delivered_packets_per_node"], dtype=np.float64
        )
        cumulative_service += delivered
        total_consumed += float(info["energy_trace"]["consumed"].sum())
        if hnd is None and info["alive"] <= env.n_nodes // 2:
            hnd = env.round
    throughput = int(env.total_packets)
    efficiency = throughput / total_consumed if total_consumed > 0.0 else 0.0
    right_censored = bool(truncated and info["alive"] > 0)
    return {
        "seed": seed,
        "policy": policy.name,
        "literature_baseline": bool(policy.literature_baseline),
        "idle_enabled": bool(idle_enabled),
        "status": "ok",
        "t_fnd": env.t_fnd,
        "t_hnd": hnd,
        "t_fnd_event_observed": env.t_fnd is not None,
        "t_hnd_event_observed": hnd is not None,
        "censor_round": env.round,
        "rounds": env.round,
        "throughput": throughput,
        "idle_energy_j": float(env.total_idle_energy),
        "queue_fairness": jain(cumulative_service),
        "energy_consumed_j": total_consumed,
        "energy_efficiency_packets_per_j": efficiency,
        "dropped_stale_packets": int(env.dropped_stale_packets),
        "dropped_overflow_packets": int(env.dropped_overflow_packets),
        "alive_at_end": int(env.alive.sum()),
        "right_censored": right_censored,
        "schedule_exhausted": bool(info.get("schedule_exhausted", False)),
    }


def summarize(rows):
    result = {}
    names = sorted({row["policy"] for row in rows})
    for name in names:
        subset = [row for row in rows if row["policy"] == name]
        metrics = {}
        for metric in METRICS:
            values = np.asarray(
                [row[metric] for row in subset if row[metric] is not None],
                dtype=np.float64,
            )
            metrics[metric] = {
                "observed": int(values.size),
                "median": float(np.median(values)) if values.size else None,
                "q1": float(np.percentile(values, 25)) if values.size else None,
                "q3": float(np.percentile(values, 75)) if values.size else None,
                "iqr": (
                    float(np.percentile(values, 75) - np.percentile(values, 25))
                    if values.size
                    else None
                ),
            }
        result[name] = metrics
    return result


def kaplan_meier_median(times, events):
    times = np.asarray(times, dtype=np.float64)
    events = np.asarray(events, dtype=bool)
    survival = 1.0
    for time in np.unique(times[events]):
        at_risk = int(np.count_nonzero(times >= time))
        deaths = int(np.count_nonzero((times == time) & events))
        if at_risk:
            survival *= 1.0 - deaths / at_risk
        if survival <= 0.5:
            return float(time)
    return None


def censor_aware_lifetime_summary(rows, reference_policy="hta_mac"):
    if not rows:
        return {}
    tau = int(min(row["censor_round"] for row in rows))
    policies = sorted({row["policy"] for row in rows})
    result = {"common_restriction_round": tau, "endpoints": {}}
    for endpoint in ("t_fnd", "t_hnd"):
        endpoint_summary = {}
        restricted_by_policy = {}
        for policy in policies:
            subset = sorted(
                (row for row in rows if row["policy"] == policy),
                key=lambda row: row["seed"],
            )
            events = np.asarray(
                [row[f"{endpoint}_event_observed"] for row in subset],
                dtype=bool,
            )
            times = np.asarray(
                [
                    row[endpoint]
                    if row[f"{endpoint}_event_observed"]
                    else row["censor_round"]
                    for row in subset
                ],
                dtype=np.float64,
            )
            restricted = np.minimum(times, tau)
            restricted_by_policy[policy] = {
                int(row["seed"]): float(value)
                for row, value in zip(subset, restricted)
            }
            endpoint_summary[policy] = {
                "trials": len(subset),
                "events": int(events.sum()),
                "right_censored": int((~events).sum()),
                "kaplan_meier_median_round": kaplan_meier_median(
                    times, events
                ),
                "median_reached": kaplan_meier_median(times, events)
                is not None,
                "restricted_mean_event_free_rounds": float(
                    restricted.mean()
                ),
                "restriction_round": tau,
                "paired_wilcoxon_vs_hta": None,
            }
        reference = restricted_by_policy.get(reference_policy, {})
        for policy in policies:
            if policy == reference_policy or not reference:
                continue
            common = sorted(
                set(reference).intersection(restricted_by_policy[policy])
            )
            reference_values = np.asarray(
                [reference[seed] for seed in common], dtype=np.float64
            )
            policy_values = np.asarray(
                [restricted_by_policy[policy][seed] for seed in common],
                dtype=np.float64,
            )
            differences = reference_values - policy_values
            if common and np.any(differences != 0.0):
                test = wilcoxon(
                    reference_values,
                    policy_values,
                    alternative="two-sided",
                    method="auto",
                )
                test_payload = {
                    "paired_trials": len(common),
                    "statistic": float(test.statistic),
                    "p_value_two_sided": float(test.pvalue),
                    "median_paired_difference_rounds": float(
                        np.median(differences)
                    ),
                }
            else:
                test_payload = {
                    "paired_trials": len(common),
                    "statistic": None,
                    "p_value_two_sided": None,
                    "median_paired_difference_rounds": (
                        float(np.median(differences))
                        if differences.size
                        else None
                    ),
                }
            endpoint_summary[policy]["paired_wilcoxon_vs_hta"] = (
                test_payload
            )
        result["endpoints"][endpoint] = endpoint_summary
    return result

def finite_row(row):
    for metric in METRICS:
        value = row[metric]
        if value is not None and not np.isfinite(float(value)):
            return False
    return True


def main():
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    set_seeds(seeds[0])
    run_dir = ROOT / "outputs" / "phase3" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    policy, solar, thermal, radio, config, manifest = build_assets(args.horizon)
    checkpoint_sha = manifest["checkpoint"]["sha256"]
    factories = phase3_policy_factories(ROOT)
    rows = []
    compatibility_rows = []
    schedule_metadata = {}
    failures = []

    for seed in seeds:
        bundle, metadata = schedule_bundle(
            policy, seed, args.horizon, checkpoint_sha
        )
        schedule_metadata[str(seed)] = metadata
        for name, factory in factories:
            try:
                row = run_one(
                    factory(), seed, bundle, solar, thermal, radio, config, True
                )
                rows.append(row)
                print(
                    f"SEED={seed} POLICY={name} FND={row['t_fnd']} "
                    f"HND={row['t_hnd']} ROUNDS={row['rounds']} "
                    f"PACKETS={row['throughput']} IDLE_J={row['idle_energy_j']:.6f}",
                    flush=True,
                )
            except Exception as exc:
                failures.append(
                    {"seed": seed, "policy": name, "error": repr(exc)}
                )
                print(
                    f"SEED={seed} POLICY={name} ERROR={exc!r}", flush=True
                )
        if not args.skip_compatibility:
            try:
                compatibility_rows.append(
                    run_one(
                        StaticEqualPolicy(),
                        seed,
                        bundle,
                        solar,
                        thermal,
                        radio,
                        config,
                        False,
                    )
                )
            except Exception as exc:
                failures.append(
                    {
                        "seed": seed,
                        "policy": "static_equal_idle_off_compatibility",
                        "error": repr(exc),
                    }
                )

    fieldnames = list(rows[0].keys()) if rows else []
    raw_csv = run_dir / "raw_trials.csv"
    if rows:
        with raw_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    compatibility_csv = run_dir / "static_idle_off_compatibility.csv"
    if compatibility_rows:
        with compatibility_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(compatibility_rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(compatibility_rows)

    expected = len(seeds) * len(factories)
    all_finite = all(finite_row(row) for row in rows)
    summary_metrics = summarize(rows)
    compatibility_summary = summarize(compatibility_rows)
    compatibility_median = None
    compatibility_relative_gap = None
    compatibility_within_20pct = None
    if compatibility_rows:
        compatibility_median = compatibility_summary["static_equal"]["t_fnd"][
            "median"
        ]
        if compatibility_median is not None:
            reproduced = 1100.6
            compatibility_relative_gap = abs(compatibility_median - reproduced) / reproduced
            compatibility_within_20pct = compatibility_relative_gap <= 0.20

    report = {
        "status": (
            "pilot_complete"
            if len(rows) == expected and not failures and all_finite
            else "pilot_failed"
        ),
        "phase3_structural_gate_pass": bool(
            len(rows) == expected and not failures and all_finite
        ),
        "inferential_superiority_claimed": False,
        "seeds": seeds,
        "horizon": args.horizon,
        "expected_primary_runs": expected,
        "completed_primary_runs": len(rows),
        "failures": failures,
        "all_metrics_finite": all_finite,
        "git_hash": git_hash(),
        "frozen_checkpoint_sha256": checkpoint_sha,
        "trained_checkpoint_sha256": file_sha256(
            ROOT
            / "outputs"
            / "phase2"
            / "authoritative_dynamic_budget8_500ep"
            / "branching_c51.pt"
        ),
        "schedule_metadata": schedule_metadata,
        "summary_median_iqr": summary_metrics,
        "censor_aware_lifetime": censor_aware_lifetime_summary(rows),
        "static_idle_off_compatibility": {
            "reference_reproduced_t_fnd": 1100.6,
            "pilot_median_t_fnd": compatibility_median,
            "relative_gap": compatibility_relative_gap,
            "within_prespecified_20_percent_diagnostic_band": compatibility_within_20pct,
            "summary": compatibility_summary,
        },
        "policy_count_resolution": (
            "six named comparison policies including HTA-MAC plus one "
            "non-literature random-budgeted diagnostic equals seven"
        ),
        "limitations": [
            "Five seeds are a development pilot and do not support inferential superiority claims.",
            "S2A2MAC HMM parameters are not published as a reusable artifact; cluster-local deterministic tertiles reproduce its three residual-energy/load layers.",
            "FFSS slot ordering cannot be represented by the round-level environment; the adaptation preserves one-slot feasible-first fixed-frame selection.",
        ],
    }
    report_path = run_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PRIMARY_RUNS={len(rows)}/{expected}")
    print(f"FAILURES={len(failures)}")
    print(f"ALL_METRICS_FINITE={all_finite}")
    print(
        "STATIC_IDLE_OFF_MEDIAN_FND="
        f"{compatibility_median} RELATIVE_GAP={compatibility_relative_gap}"
    )
    print(f"PHASE3_STRUCTURAL_GATE_PASS={report['phase3_structural_gate_pass']}")
    print(f"report={report_path}")
    return 0 if report["phase3_structural_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
