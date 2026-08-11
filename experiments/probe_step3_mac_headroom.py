"""Cheap no-learning test of whether MAC allocation has usable headroom."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.interface import MACPolicyInterface
from baselines.policies import EnergyProportionalPolicy
from experiments.paper_aligned_environment import (
    configure_mac,
    disabled_thermal_hmm,
    load_profile,
    schedule_bundle,
)
from experiments.run_phase3_pilot import build_assets, run_one


def capped_allocate(scores, caps, budget):
    allocation = np.zeros(len(scores), dtype=np.int64)
    for _ in range(int(budget)):
        eligible = allocation < caps
        if not np.any(eligible):
            break
        quotient = np.full(len(scores), -np.inf)
        quotient[eligible] = scores[eligible] / (allocation[eligible] + 1.0)
        selected = int(np.argmax(quotient))
        if not np.isfinite(quotient[selected]):
            break
        allocation[selected] += 1
    return allocation


class CHConditionalConservativePolicy(MACPolicyInterface):
    def __init__(self, reserve_threshold, low_reserve_budget_fraction):
        self.reserve_threshold = float(reserve_threshold)
        self.low_reserve_budget_fraction = float(low_reserve_budget_fraction)
        self.name = f"ch_protect_r{reserve_threshold:.2f}_b{low_reserve_budget_fraction:.2f}"

    def select_action(self, state, env):
        action = np.zeros(env.n_nodes, dtype=np.int64)
        for cluster, ch in enumerate(env.cluster_heads):
            ch = int(ch)
            members = self.eligible_members(env, cluster, ch)
            if not len(members):
                continue
            reserve = float(env.energy[ch] / env.cfg.initial_energy_j)
            budget = env.cfg.frame_slot_budget
            if reserve < self.reserve_threshold:
                budget = max(1, int(np.ceil(budget * self.low_reserve_budget_fraction)))
            caps = np.minimum(env.queue[members], env.cfg.n_max).astype(np.int64)
            ages = np.asarray([
                max(env.packet_ages[node]) if env.packet_ages[node] else 0
                for node in members
            ], dtype=np.float64)
            age_score = ages / max(1, env.cfg.packet_ttl_rounds)
            queue_score = caps / max(1, env.cfg.n_max)
            energy_score = np.clip(env.energy[members] / env.cfg.initial_energy_j, 0.0, 1.0)
            scores = 0.50 * age_score + 0.30 * queue_score + 0.20 * energy_score
            action[members] = capped_allocate(scores, caps, budget)
        return self.validate(action, env)


def lifetime_value(row):
    return float(row["t_fnd"] if row["t_fnd"] is not None else row["censor_round"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-profile", type=Path, required=True)
    parser.add_argument("--seeds", default="2400,2401,2402,2403,2404")
    parser.add_argument("--horizon", type=int, default=1200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value]
    if seeds != [2400, 2401, 2402, 2403, 2404]:
        raise ValueError("headroom gate requires frozen development seeds 2400-2404")
    if args.horizon < 1000:
        raise ValueError("headroom horizon must cover at least 1000 rounds")
    profile_path = args.environment_profile if args.environment_profile.is_absolute() else ROOT / args.environment_profile
    profile, profile_evidence = load_profile(profile_path)
    _, solar, thermal, radio, config, _ = build_assets(args.horizon)
    thermal = disabled_thermal_hmm(thermal)
    config = configure_mac(profile, config)
    policies = [EnergyProportionalPolicy(score_exponent=2.0)] + [
        CHConditionalConservativePolicy(reserve, fraction)
        for reserve in (0.20, 0.30, 0.40)
        for fraction in (0.55, 0.70)
    ]
    rows = []
    for seed in seeds:
        bundle, _ = schedule_bundle(profile, solar, seed, args.horizon)
        for policy in policies:
            rows.append(run_one(policy, seed, bundle, solar, thermal, radio, config, False))
    reference = {int(row["seed"]): row for row in rows if row["policy"] == "energy_proportional"}
    candidates = []
    for policy in policies[1:]:
        subset = {int(row["seed"]): row for row in rows if row["policy"] == policy.name}
        fnd_delta = [lifetime_value(subset[s]) - lifetime_value(reference[s]) for s in seeds]
        efficiency_delta = [
            subset[s]["energy_efficiency_packets_per_j"] / max(reference[s]["energy_efficiency_packets_per_j"], 1e-12) - 1.0
            for s in seeds
        ]
        qos = [
            subset[s]["delivery_ratio"] >= 0.55
            and subset[s]["stale_drop_ratio"] <= 0.45
            and subset[s]["queue_fairness"] >= 0.70
            for s in seeds
        ]
        median_fnd = float(np.median(fnd_delta))
        median_efficiency = float(np.median(efficiency_delta))
        lifetime_signal = median_fnd >= 0.02 * args.horizon
        efficiency_signal = median_fnd >= -0.01 * args.horizon and median_efficiency >= 0.03
        candidates.append({
            "policy": policy.name,
            "all_five_qos_pass": all(qos),
            "paired_fnd_or_censor_delta_rounds": fnd_delta,
            "median_fnd_or_censor_delta_rounds": median_fnd,
            "paired_packets_per_joule_relative_delta": efficiency_delta,
            "median_packets_per_joule_relative_delta": median_efficiency,
            "lifetime_signal": lifetime_signal,
            "efficiency_signal_at_lifetime_noninferiority": efficiency_signal,
            "headroom_pass": bool(all(qos) and (lifetime_signal or efficiency_signal)),
        })
    passed = [row for row in candidates if row["headroom_pass"]]
    payload = {
        "schema_version": 1,
        "status": "mac_headroom_demonstrated" if passed else "mac_headroom_not_demonstrated_stop_training",
        "overall_pass": bool(passed),
        "development_only": True,
        "profile_sha256": profile_evidence["sha256"],
        "seeds": seeds,
        "horizon": args.horizon,
        "reference": "energy_proportional",
        "thresholds": {
            "all_five_delivery_min": 0.55,
            "all_five_stale_max": 0.45,
            "all_five_fairness_min": 0.70,
            "lifetime_signal_rounds": 0.02 * args.horizon,
            "lifetime_noninferiority_margin_rounds": 0.01 * args.horizon,
            "packets_per_joule_relative_improvement": 0.03,
        },
        "candidates": candidates,
        "passing_candidates": [row["policy"] for row in passed],
        "raw_trials": rows,
        "claim_boundary": "diagnostic_reachability_gate_not_model_performance",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "passing_candidates": payload["passing_candidates"]}, indent=2))
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__":
    main()
