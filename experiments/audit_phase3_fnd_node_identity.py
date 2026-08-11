"""Replay paired policies to identify the physical node(s) causing FND.

This is a development diagnostic, not an outcome or superiority test.  It uses
the same environment construction, frozen schedules, and policy factories as
``run_phase3_pilot.py`` and verifies replayed FND rounds against archived rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines import phase3_policy_factories
from envs.scheduled_mac_env import ScheduledIntraClusterMACEnv
from experiments.paper_aligned_environment import (
    configure_mac as configure_paper_aligned_mac,
    disabled_thermal_hmm,
    load_profile as load_environment_profile,
    schedule_bundle as paper_aligned_schedule_bundle,
)
from experiments.run_phase3_pilot import build_assets, set_seeds


DEFAULT_PROFILE = ROOT / "config" / "paper_aligned_hasani2025_b16_qos_repaired.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archived-raw-csv", type=Path)
    parser.add_argument("--environment-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--seeds", default="2400,2401,2402,2403,2404")
    parser.add_argument("--policies", default="hta_mac,energy_proportional")
    parser.add_argument("--horizon", type=int, default=3000)
    parser.add_argument("--hta-budget", type=int, default=16)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--expected-environment-profile-sha256")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def archived_fnd_rows(path: Path | None) -> dict[tuple[int, str], int]:
    if path is None:
        return {}
    with resolve_path(path).open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return {
            (int(row["seed"]), row["policy"]): int(float(row["t_fnd"]))
            for row in rows
            if row.get("t_fnd") not in (None, "")
        }


def run_to_fnd(policy, seed, bundle, solar, thermal, radio, config):
    """Run one policy until FND and retain the death-transition evidence."""
    env = ScheduledIntraClusterMACEnv(
        config, radio, solar, thermal, idle_energy_enabled=False
    )
    policy.reset(seed)
    state, _ = env.reset(seed=seed, frozen_snapshot=bundle)
    ch_assignments = np.zeros(env.n_nodes, dtype=np.int64)
    cumulative_consumed = np.zeros(env.n_nodes, dtype=np.float64)
    cumulative_harvested = np.zeros(env.n_nodes, dtype=np.float64)
    cumulative_slots = np.zeros(env.n_nodes, dtype=np.int64)
    cumulative_delivered = np.zeros(env.n_nodes, dtype=np.int64)

    while env.t_fnd is None:
        alive_before = env.alive.copy()
        energy_before = env.energy.copy()
        queue_before = env.queue.copy()
        heads_before = env.cluster_heads.copy()
        cluster_before = env.cluster_of.copy()
        positions_before = env.positions.copy()
        ch_assignments[heads_before] += 1

        action = np.asarray(policy.select_action(state, env), dtype=np.int64)
        state, _, terminated, truncated, info = env.step(action)
        trace = info["energy_trace"]
        delivered = np.asarray(info["delivered_packets_per_node"], dtype=np.int64)
        cumulative_consumed += np.asarray(trace["consumed"], dtype=np.float64)
        cumulative_harvested += np.asarray(trace["harvested"], dtype=np.float64)
        cumulative_slots += action
        cumulative_delivered += delivered
        newly_dead = np.flatnonzero(alive_before & ~env.alive)

        if newly_dead.size:
            records = []
            head_set = {int(value) for value in heads_before}
            for node_value in newly_dead:
                node = int(node_value)
                cluster = int(cluster_before[node])
                ch = int(heads_before[cluster])
                records.append(
                    {
                        "node_id": node,
                        "position_m": positions_before[node].tolist(),
                        "was_ch_on_fnd_round": node in head_set,
                        "cluster_on_fnd_round": cluster,
                        "scheduled_ch_on_fnd_round": ch,
                        "energy_before_j": float(energy_before[node]),
                        "energy_after_j": float(env.energy[node]),
                        "consumed_on_fnd_round_j": float(trace["consumed"][node]),
                        "harvested_on_fnd_round_j": float(trace["harvested"][node]),
                        "queue_before": int(queue_before[node]),
                        "allocated_slots_on_fnd_round": int(action[node]),
                        "delivered_on_fnd_round": int(delivered[node]),
                        "ch_assignments_through_fnd": int(ch_assignments[node]),
                        "ch_assignment_fraction_through_fnd": float(
                            ch_assignments[node] / env.round
                        ),
                        "cumulative_consumed_j": float(cumulative_consumed[node]),
                        "cumulative_harvested_j": float(cumulative_harvested[node]),
                        "cumulative_allocated_slots": int(cumulative_slots[node]),
                        "cumulative_delivered_packets": int(cumulative_delivered[node]),
                        "distance_to_bs_m": float(
                            np.linalg.norm(
                                positions_before[node]
                                - np.asarray(config.bs_position_m, dtype=np.float64)
                            )
                        ),
                    }
                )
            return {
                "seed": int(seed),
                "policy": policy.name,
                "fnd_round": int(env.t_fnd),
                "newly_dead_count": int(newly_dead.size),
                "newly_dead_nodes": records,
            }

        if terminated or truncated:
            raise RuntimeError(
                f"FND not observed for seed={seed} policy={policy.name} "
                f"before round {env.round}"
            )


def summarize(records: list[dict], policies: list[str]) -> dict:
    by_policy = {}
    for policy in policies:
        nodes = [
            node["node_id"]
            for record in records
            if record["policy"] == policy
            for node in record["newly_dead_nodes"]
        ]
        counts = Counter(nodes)
        by_policy[policy] = {
            "trials": sum(record["policy"] == policy for record in records),
            "unique_fnd_node_ids": sorted(counts),
            "fnd_node_frequency": {
                str(node): int(counts[node]) for node in sorted(counts)
            },
            "maximum_single_node_frequency": max(counts.values(), default=0),
        }

    paired = []
    if len(policies) == 2:
        left, right = policies
        for seed in sorted({record["seed"] for record in records}):
            left_nodes = {
                node["node_id"]
                for record in records
                if record["seed"] == seed and record["policy"] == left
                for node in record["newly_dead_nodes"]
            }
            right_nodes = {
                node["node_id"]
                for record in records
                if record["seed"] == seed and record["policy"] == right
                for node in record["newly_dead_nodes"]
            }
            paired.append(
                {
                    "seed": int(seed),
                    "left_policy": left,
                    "left_nodes": sorted(left_nodes),
                    "right_policy": right,
                    "right_nodes": sorted(right_nodes),
                    "same_fnd_node_set": left_nodes == right_nodes,
                    "any_fnd_node_overlap": bool(left_nodes.intersection(right_nodes)),
                }
            )
    same_count = sum(row["same_fnd_node_set"] for row in paired)
    overlap_count = sum(row["any_fnd_node_overlap"] for row in paired)
    return {
        "by_policy": by_policy,
        "paired_seed_comparison": paired,
        "paired_same_node_set_count": int(same_count),
        "paired_any_node_overlap_count": int(overlap_count),
        "paired_trial_count": len(paired),
        "same_node_set_fraction": same_count / len(paired) if paired else None,
        "any_node_overlap_fraction": overlap_count / len(paired) if paired else None,
    }


def main() -> None:
    args = parse_args()
    checkpoint = resolve_path(args.checkpoint)
    output = resolve_path(args.output)
    profile_path = resolve_path(args.environment_profile)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    policy_names = [value.strip() for value in args.policies.split(",") if value.strip()]
    if len(policy_names) != 2:
        raise ValueError("this paired diagnostic requires exactly two policies")

    checkpoint_sha = file_sha256(checkpoint)
    profile_sha = file_sha256(profile_path)
    if args.expected_checkpoint_sha256:
        assert checkpoint_sha == args.expected_checkpoint_sha256.lower()
    if args.expected_environment_profile_sha256:
        assert profile_sha == args.expected_environment_profile_sha256.lower()

    set_seeds(seeds[0])
    _, solar, thermal, radio, config, _ = build_assets(args.horizon)
    profile, profile_evidence = load_environment_profile(profile_path)
    prohibited = set(profile["prohibited_registered_held_out_seeds"])
    reserved = set(profile["reserved_confirmation_seeds"])
    development = set(profile["development_seeds"])
    if set(seeds) - development:
        raise ValueError("FND identity audit is restricted to development seeds")
    if set(seeds).intersection(prohibited | reserved):
        raise ValueError("held-out or confirmation seed requested")
    thermal = disabled_thermal_hmm(thermal)
    config = configure_paper_aligned_mac(profile, config)

    factory_map = dict(
        phase3_policy_factories(
            ROOT, hta_checkpoint=checkpoint, hta_budget=args.hta_budget
        )
    )
    missing = set(policy_names) - set(factory_map)
    if missing:
        raise ValueError(f"unknown policies: {sorted(missing)}")
    archived = archived_fnd_rows(args.archived_raw_csv)

    records = []
    replay_checks = []
    for seed in seeds:
        bundle, metadata = paper_aligned_schedule_bundle(
            profile, solar, seed, args.horizon
        )
        for policy_name in policy_names:
            record = run_to_fnd(
                factory_map[policy_name](),
                seed,
                bundle,
                solar,
                thermal,
                radio,
                config,
            )
            expected = archived.get((seed, policy_name))
            match = expected is None or record["fnd_round"] == expected
            replay_checks.append(
                {
                    "seed": seed,
                    "policy": policy_name,
                    "archived_fnd_round": expected,
                    "replayed_fnd_round": record["fnd_round"],
                    "match": bool(match),
                }
            )
            if not match:
                raise RuntimeError(
                    f"FND replay mismatch for seed={seed} policy={policy_name}: "
                    f"expected {expected}, observed {record['fnd_round']}"
                )
            record["schedule_stop_reason"] = metadata.get("stop_reason")
            records.append(record)
            print(
                f"seed={seed} policy={policy_name} fnd={record['fnd_round']} "
                f"nodes={[node['node_id'] for node in record['newly_dead_nodes']]}",
                flush=True,
            )

    summary = summarize(records, policy_names)
    payload = {
        "schema_version": 1,
        "status": "development_fnd_node_identity_audit_complete",
        "interpretation": (
            "mechanism diagnostic on frozen development trials; not an outcome "
            "or superiority test"
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "environment_profile": profile_evidence,
        "environment_profile_sha256": profile_sha,
        "run_phase3_pilot_sha256": file_sha256(
            ROOT / "experiments" / "run_phase3_pilot.py"
        ),
        "seeds": seeds,
        "policies": policy_names,
        "horizon": args.horizon,
        "held_out_or_confirmation_seeds_used": False,
        "archived_fnd_replay_checks": replay_checks,
        "all_archived_fnd_rounds_reproduced": all(
            row["match"] for row in replay_checks
        ),
        "summary": summary,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"output={output}")


if __name__ == "__main__":
    main()
