"""Second cheap headroom test: preserve energy ranking and vary only CH-conditioned budget."""

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
from experiments.paper_aligned_environment import configure_mac, disabled_thermal_hmm, load_profile, schedule_bundle
from experiments.probe_step3_mac_headroom import capped_allocate, lifetime_value
from experiments.run_phase3_pilot import build_assets, run_one


class CHConditionalEnergyPolicy(MACPolicyInterface):
    def __init__(self, reserve_threshold, low_reserve_budget_fraction):
        self.reserve_threshold = float(reserve_threshold)
        self.low_reserve_budget_fraction = float(low_reserve_budget_fraction)
        self.name = f"ch_energy_rank_r{reserve_threshold:.2f}_b{low_reserve_budget_fraction:.2f}"

    def select_action(self, state, env):
        action = np.zeros(env.n_nodes, dtype=np.int64)
        for cluster, ch in enumerate(env.cluster_heads):
            ch = int(ch)
            members = self.eligible_members(env, cluster, ch)
            if not len(members):
                continue
            budget = env.cfg.frame_slot_budget
            if env.energy[ch] / env.cfg.initial_energy_j < self.reserve_threshold:
                budget = max(1, int(np.ceil(budget * self.low_reserve_budget_fraction)))
            caps = np.minimum(env.queue[members], env.cfg.n_max).astype(np.int64)
            scores = np.square(np.clip(env.energy[members] / env.cfg.initial_energy_j, 0.0, 1.0))
            action[members] = capped_allocate(scores, caps, budget)
        return self.validate(action, env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-profile", type=Path, required=True)
    parser.add_argument("--seeds", default="2400,2401,2402,2403,2404")
    parser.add_argument("--horizon", type=int, default=1200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value]
    if seeds != [2400, 2401, 2402, 2403, 2404] or args.horizon < 1000:
        raise ValueError("use frozen seeds 2400-2404 and horizon >=1000")
    profile_path = args.environment_profile if args.environment_profile.is_absolute() else ROOT / args.environment_profile
    profile, evidence = load_profile(profile_path)
    _, solar, thermal, radio, config, _ = build_assets(args.horizon)
    thermal, config = disabled_thermal_hmm(thermal), configure_mac(profile, config)
    policies = [EnergyProportionalPolicy(score_exponent=2.0)] + [
        CHConditionalEnergyPolicy(reserve, fraction)
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
        fnd = [lifetime_value(subset[s]) - lifetime_value(reference[s]) for s in seeds]
        efficiency = [subset[s]["energy_efficiency_packets_per_j"] / reference[s]["energy_efficiency_packets_per_j"] - 1.0 for s in seeds]
        qos = [subset[s]["delivery_ratio"] >= 0.55 and subset[s]["stale_drop_ratio"] <= 0.45 and subset[s]["queue_fairness"] >= 0.70 for s in seeds]
        median_fnd, median_efficiency = float(np.median(fnd)), float(np.median(efficiency))
        passed = bool(all(qos) and (median_fnd >= 0.02 * args.horizon or (median_fnd >= -0.01 * args.horizon and median_efficiency >= 0.03)))
        candidates.append({
            "policy": policy.name, "all_five_qos_pass": all(qos),
            "paired_fnd_or_censor_delta_rounds": fnd,
            "median_fnd_or_censor_delta_rounds": median_fnd,
            "paired_packets_per_joule_relative_delta": efficiency,
            "median_packets_per_joule_relative_delta": median_efficiency,
            "headroom_pass": passed,
        })
    passing = [row for row in candidates if row["headroom_pass"]]
    payload = {
        "schema_version": 1,
        "status": "mac_headroom_demonstrated" if passing else "mac_headroom_not_demonstrated_stop_training",
        "overall_pass": bool(passing), "profile_sha256": evidence["sha256"],
        "seeds": seeds, "horizon": args.horizon, "reference": "energy_proportional",
        "candidates": candidates, "passing_candidates": [row["policy"] for row in passing],
        "raw_trials": rows,
        "claim_boundary": "second_development_headroom_diagnostic_not_model_performance",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "passing_candidates": payload["passing_candidates"]}, indent=2))
    raise SystemExit(0 if passing else 3)


if __name__ == "__main__":
    main()
