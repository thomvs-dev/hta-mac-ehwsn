"""Paired mid-episode hybrid-trajectory policy sensitivity diagnostic.

This development-only audit evaluates two checkpoints on identical states with
queue caps >= 2. It changes the probed node from the lowest joint solar/thermal
trajectory block to the highest block while holding energy and all other node
features fixed. It is not a Phase 2 gate or held-out performance evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.hmm.rectified_moments import next_rectified_statistics
from experiments.audit_phase2_trajectory_sensitivity import (
    build_curriculum,
    load_agent,
)
from experiments.train_phase2_dynamic_curriculum import padded_state


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("original", type=Path)
    parser.add_argument("repaired", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-seeds", default="2300,2301,2302,2303,2304")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--rollout-steps", type=int, default=60)
    parser.add_argument("--nodes-per-state", type=int, default=3)
    parser.add_argument("--rollout-epsilon", type=float, default=0.15)
    parser.add_argument("--audit-seed", type=int, default=20260803)
    return parser.parse_args()


def hybrid_pair(env, state, node):
    solar_mean, solar_var = next_rectified_statistics(
        env.base.solar.transition,
        env.base.solar.mean,
        env.base.solar.variance,
        env.base.cfg.solar_scale,
    )
    thermal_mean, thermal_var = next_rectified_statistics(
        env.base.thermal.transition,
        env.base.thermal.mean,
        env.base.thermal.variance,
        env.base.cfg.thermal_scale,
    )
    low, high = state.copy(), state.copy()
    low[node, 0] = high[node, 0] = 0.5
    low[node, 1:3] = (
        solar_mean[0] + thermal_mean[0],
        solar_var[0] + thermal_var[0],
    )
    high[node, 1:3] = (
        solar_mean[-1] + thermal_mean[-1],
        solar_var[-1] + thermal_var[-1],
    )
    low[node, 3:11] = env.base.solar.transition[0]
    high[node, 3:11] = env.base.solar.transition[-1]
    low[node, 11:15] = env.base.thermal.transition[0]
    high[node, 11:15] = env.base.thermal.transition[-1]
    return low, high


def collect_probes(environments, rollout_agent, args):
    probes = []
    for env in environments:
        observation, mask, _ = env.reset()
        for step in range(args.rollout_steps):
            state, active_mask, caps = padded_state(
                env, observation, mask, env.base.n_nodes
            )
            eligible = np.flatnonzero(active_mask & (caps >= 2))
            for node in eligible[: args.nodes_per_state]:
                low, high = hybrid_pair(env, state, int(node))
                probes.append(
                    {
                        "low": low,
                        "high": high,
                        "mask": active_mask.copy(),
                        "caps": caps.copy(),
                        "node": int(node),
                        "seed": int(env.seed),
                        "target_rank": int(env.target_rank),
                        "round": int(step),
                    }
                )
            action, _ = rollout_agent.act(
                state,
                active_mask,
                epsilon=args.rollout_epsilon,
                caps=caps,
            )
            observation, mask, done, _ = env.step(action)
            if done:
                break
    return probes


def inspect(agent, probes):
    counters = {
        "local_argmax_changes": 0,
        "projected_node_changes": 0,
        "projected_vector_changes": 0,
        "high_more_slots": 0,
        "high_fewer_slots": 0,
        "all_marginals_ordered": 0,
    }
    for probe in probes:
        states = torch.as_tensor(
            np.stack((probe["low"], probe["high"])), dtype=torch.float32
        )
        masks = torch.as_tensor(
            np.stack((probe["mask"], probe["mask"])), dtype=torch.bool
        )
        with torch.no_grad():
            q_values = agent.q_values_tensor(states, masks).cpu().numpy()
        node = probe["node"]
        low_action = agent._project(
            q_values[0], probe["mask"], caps=probe["caps"]
        )
        high_action = agent._project(
            q_values[1], probe["mask"], caps=probe["caps"]
        )
        low_marginal = np.diff(q_values[0, node])
        high_marginal = np.diff(q_values[1, node])
        counters["local_argmax_changes"] += int(
            np.argmax(q_values[0, node]) != np.argmax(q_values[1, node])
        )
        counters["projected_node_changes"] += int(
            low_action[node] != high_action[node]
        )
        counters["projected_vector_changes"] += int(
            not np.array_equal(low_action, high_action)
        )
        counters["high_more_slots"] += int(
            high_action[node] > low_action[node]
        )
        counters["high_fewer_slots"] += int(
            high_action[node] < low_action[node]
        )
        counters["all_marginals_ordered"] += int(
            np.all(high_marginal >= low_marginal)
        )
    counters["probe_count"] = len(probes)
    for name in tuple(counters):
        if name != "probe_count":
            counters[f"{name}_fraction"] = (
                counters[name] / len(probes) if probes else 0.0
            )
    return counters


def main():
    args = parse_args()
    np.random.seed(args.audit_seed)
    torch.manual_seed(args.audit_seed)
    seeds = [int(value) for value in args.development_seeds.split(",")]
    environments, _, _ = build_curriculum(seeds, args.max_steps)
    original = load_agent(args.original)
    repaired = load_agent(args.repaired)
    probes = collect_probes(environments, repaired, args)
    report = {
        "status": "development_diagnostic_not_phase2_gate",
        "counterfactual": (
            "same node and energy; joint lowest versus highest solar+thermal "
            "rectified moments and state-conditioned transition probabilities"
        ),
        "eligibility": "active member with queue-derived action cap >= 2",
        "paired_identical_states": True,
        "development_seeds": seeds,
        "audit_seed": args.audit_seed,
        "rollout_policy": str(args.repaired.resolve()),
        "rollout_epsilon": args.rollout_epsilon,
        "original_checkpoint": str(args.original.resolve()),
        "repaired_checkpoint": str(args.repaired.resolve()),
        "original": inspect(original, probes),
        "repaired": inspect(repaired, probes),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()