"""Audit policy decisions on identical HTA-MAC-controlled state trajectories.

This is a development diagnostic, not an outcome comparison. Every comparator
is queried on the same state before the environment advances with HTA-MAC's
action, avoiding the ambiguity caused by comparing actions from diverged states.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from baselines import phase3_policy_factories
from envs.scheduled_mac_env import ScheduledIntraClusterMACEnv
from experiments.paper_aligned_environment import (
    configure_mac,
    disabled_thermal_hmm,
    load_profile,
    schedule_bundle,
)
from experiments.run_phase3_pilot import build_assets, set_seeds




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--environment-profile",
        default="config/paper_aligned_hasani2025_b16_qos_repaired.json",
    )
    parser.add_argument("--seeds", default="2400,2401,2402,2403,2404")
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--hta-budget", type=int, default=16)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _jaccard(first, second):
    first = set(np.flatnonzero(first > 0).tolist())
    second = set(np.flatnonzero(second > 0).tolist())
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def main():
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    profile_path = Path(args.environment_profile)
    if not profile_path.is_absolute():
        profile_path = ROOT / profile_path
    profile, profile_evidence = load_profile(profile_path)
    if not set(seeds).issubset(profile["development_seeds"]):
        raise ValueError("action audit is restricted to declared development seeds")
    forbidden = set(seeds) & set(profile["prohibited_registered_held_out_seeds"])
    if forbidden:
        raise ValueError(f"registered held-out seeds are forbidden: {sorted(forbidden)}")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    set_seeds(seeds[0])
    _, solar, thermal, radio, base_config, _ = build_assets(args.horizon)
    thermal = disabled_thermal_hmm(thermal)
    config = configure_mac(profile, base_config)
    factories = phase3_policy_factories(
        ROOT, hta_checkpoint=checkpoint, hta_budget=args.hta_budget
    )

    aggregates = {}
    per_seed = []
    for seed in seeds:
        bundle, _ = schedule_bundle(profile, solar, seed, args.horizon)
        env = ScheduledIntraClusterMACEnv(
            config, radio, solar, thermal, idle_energy_enabled=False
        )
        policies = {name: factory() for name, factory in factories}
        for policy in policies.values():
            policy.reset(seed)
        state, _ = env.reset(seed=seed, frozen_snapshot=bundle)
        rows = {name: [] for name in policies if name != "hta_mac"}
        signatures = {name: hashlib.sha256() for name in policies}
        terminated = truncated = False
        rounds = 0
        while not terminated and not truncated:
            actions = {
                name: policy.select_action(state, env)
                for name, policy in policies.items()
            }
            reference = actions["hta_mac"]
            for name, action in actions.items():
                signatures[name].update(np.asarray(action, dtype=np.int16).tobytes())
                if name == "hta_mac":
                    continue
                scale = max(1, int(reference.sum()), int(action.sum()))
                rows[name].append(
                    {
                        "exact": bool(np.array_equal(reference, action)),
                        "normalized_l1": float(np.abs(reference - action).sum() / scale),
                        "active_jaccard": float(_jaccard(reference, action)),
                        "hta_slots": int(reference.sum()),
                        "other_slots": int(action.sum()),
                    }
                )
            state, _, terminated, truncated, _ = env.step(reference)
            rounds += 1

        seed_summary = {"seed": seed, "rounds": rounds, "comparisons": {}}
        for name, observations in rows.items():
            exact = np.asarray([row["exact"] for row in observations], dtype=float)
            l1 = np.asarray([row["normalized_l1"] for row in observations])
            jaccard = np.asarray([row["active_jaccard"] for row in observations])
            seed_summary["comparisons"][name] = {
                "round_action_agreement_fraction": float(exact.mean()),
                "mean_normalized_l1": float(l1.mean()),
                "mean_active_set_jaccard": float(jaccard.mean()),
                "hta_mean_slots": float(np.mean([row["hta_slots"] for row in observations])),
                "other_mean_slots": float(np.mean([row["other_slots"] for row in observations])),
                "identical_trajectory_signature": (
                    signatures[name].hexdigest() == signatures["hta_mac"].hexdigest()
                ),
            }
            aggregates.setdefault(name, []).extend(observations)
        per_seed.append(seed_summary)

    aggregate = {}
    for name, observations in aggregates.items():
        aggregate[name] = {
            "observations": len(observations),
            "round_action_agreement_fraction": float(np.mean([row["exact"] for row in observations])),
            "mean_normalized_l1": float(np.mean([row["normalized_l1"] for row in observations])),
            "mean_active_set_jaccard": float(np.mean([row["active_jaccard"] for row in observations])),
        }
    payload = {
        "schema_version": 1,
        "status": "development_action_distinctness_audit",
        "reference_policy": "hta_mac",
        "trajectory_controller": "hta_mac",
        "interpretation": "decision comparison on common states; not an outcome or superiority test",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "environment_profile": profile_evidence,
        "seeds": seeds,
        "horizon": args.horizon,
        "aggregate": aggregate,
        "per_seed": per_seed,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    print(f"WROTE {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
