"""Post-hoc policy-level trajectory sensitivity audit for registered Phase 2.

This diagnostic is deliberately separate from the frozen Phase 2 gate.  It
tests whether changing one active node from the solar S1 transition profile to
the S8 profile changes its greedy local decision or the budget-projected joint
allocation across many development-curriculum states.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.branching_dqn import BranchingAgentConfig, BranchingDQNAgent
from core.hmm.rectified_moments import next_rectified_statistics
from experiments.train_phase2_dynamic_curriculum import build_curriculum


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values) if values else math.nan,
        "q1": percentile(values, 0.25),
        "q3": percentile(values, 0.75),
        "minimum": min(values) if values else math.nan,
        "maximum": max(values) if values else math.nan,
    }


def select_nodes(active: np.ndarray, maximum: int) -> np.ndarray:
    if active.size <= maximum:
        return active
    positions = np.linspace(0, active.size - 1, maximum).round().astype(int)
    return active[positions]


def load_agent(checkpoint: Path) -> BranchingDQNAgent:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config_data = dict(payload["config"])
    config_data["precision"] = "fp32"
    config = BranchingAgentConfig(**config_data)
    agent = BranchingDQNAgent(config, device="cpu")
    agent.online.load_state_dict(payload["online_state_dict"])
    agent.online.eval()
    return agent


def build_probes(environments, nodes_per_environment: int):
    probes = []
    empty_environments = 0
    for environment_index, env in enumerate(environments):
        observation, mask, _ = env.reset()
        active = np.flatnonzero(mask)
        if not active.size:
            empty_environments += 1
            continue
        solar_mean, solar_variance = next_rectified_statistics(
            env.base.solar.transition,
            env.base.solar.mean,
            env.base.solar.variance,
            env.base.cfg.solar_scale,
        )
        caps = np.minimum(env.base.queue, env.base.cfg.n_max).astype(np.int64)
        caps[~mask] = 0
        for node in select_nodes(active, nodes_per_environment):
            low = observation.copy()
            high = observation.copy()
            low[node, 0] = high[node, 0] = 0.5
            low[node, 3:11] = env.base.solar.transition[0]
            high[node, 3:11] = env.base.solar.transition[7]
            low[node, 1], low[node, 2] = solar_mean[0], solar_variance[0]
            high[node, 1], high[node, 2] = solar_mean[7], solar_variance[7]
            probes.append(
                {
                    "environment_index": environment_index,
                    "seed": env.seed,
                    "target_rank": env.target_rank,
                    "node": int(node),
                    "low": low.astype(np.float32),
                    "high": high.astype(np.float32),
                    "mask": mask.astype(bool),
                    "caps": caps,
                }
            )
    return probes, empty_environments


def inspect_run(run_dir: Path, probes: list[dict]) -> dict:
    agent = load_agent(run_dir / "branching_c51.pt")
    states = np.stack(
        [state for probe in probes for state in (probe["low"], probe["high"])]
    )
    masks = np.stack(
        [probe["mask"] for probe in probes for _ in range(2)]
    )
    with torch.no_grad():
        q_values = agent.q_values_tensor(
            torch.as_tensor(states, dtype=torch.float32),
            torch.as_tensor(masks, dtype=torch.bool),
        ).cpu().numpy()

    records = []
    for index, probe in enumerate(probes):
        low_q_all = q_values[2 * index]
        high_q_all = q_values[2 * index + 1]
        node = probe["node"]
        low_q = low_q_all[node]
        high_q = high_q_all[node]
        scale = max(float(np.max(np.abs(low_q))), float(np.max(np.abs(high_q))), 1e-12)
        low_local = int(np.argmax(low_q))
        high_local = int(np.argmax(high_q))
        low_action = agent._project(low_q_all, probe["mask"], caps=probe["caps"])
        high_action = agent._project(high_q_all, probe["mask"], caps=probe["caps"])
        records.append(
            {
                "environment_index": probe["environment_index"],
                "seed": probe["seed"],
                "target_rank": probe["target_rank"],
                "node": node,
                "max_absolute_q_difference": float(np.max(np.abs(high_q - low_q))),
                "relative_q_difference": float(np.max(np.abs(high_q - low_q)) / scale),
                "local_argmax_changed": low_local != high_local,
                "local_s1_argmax": low_local,
                "local_s8_argmax": high_local,
                "projected_node_changed": int(low_action[node]) != int(high_action[node]),
                "projected_vector_changed": not np.array_equal(low_action, high_action),
                "projected_s1_node_slots": int(low_action[node]),
                "projected_s8_node_slots": int(high_action[node]),
                "s8_strictly_more_slots": int(high_action[node]) > int(low_action[node]),
                "s8_fewer_slots": int(high_action[node]) < int(low_action[node]),
            }
        )

    count = len(records)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "run_name": run_dir.name,
        "architecture": summary["agent_config"]["architecture"],
        "budget": summary["projection_budget"],
        "optimizer_seed": summary["optimizer_seed"],
        "probe_count": count,
        "relative_q_difference": distribution(
            [record["relative_q_difference"] for record in records]
        ),
        "absolute_q_difference": distribution(
            [record["max_absolute_q_difference"] for record in records]
        ),
        "local_argmax_change_count": sum(record["local_argmax_changed"] for record in records),
        "local_argmax_change_fraction": sum(record["local_argmax_changed"] for record in records) / count,
        "projected_node_change_count": sum(record["projected_node_changed"] for record in records),
        "projected_node_change_fraction": sum(record["projected_node_changed"] for record in records) / count,
        "projected_vector_change_count": sum(record["projected_vector_changed"] for record in records),
        "projected_vector_change_fraction": sum(record["projected_vector_changed"] for record in records) / count,
        "s8_strictly_more_slots_count": sum(record["s8_strictly_more_slots"] for record in records),
        "s8_fewer_slots_count": sum(record["s8_fewer_slots"] for record in records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-seeds", default="2300,2301,2302,2303,2304")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--nodes-per-environment", type=int, default=5)
    args = parser.parse_args()

    seeds = [int(value) for value in args.development_seeds.split(",")]
    environments, manifest, _ = build_curriculum(seeds, args.max_steps)
    probes, empty_environments = build_probes(environments, args.nodes_per_environment)
    if not probes:
        raise RuntimeError("no active-node probes were generated")

    run_dirs = sorted(path for path in (args.artifact / "runs").iterdir() if path.is_dir())
    runs = [inspect_run(run_dir, probes) for run_dir in run_dirs]
    report = {
        "diagnostic_status": "post_hoc_development_diagnostic_not_original_gate",
        "artifact": str(args.artifact.resolve()),
        "development_seeds": seeds,
        "max_steps": args.max_steps,
        "curriculum_environment_count": len(environments),
        "empty_environment_count": empty_environments,
        "nodes_per_environment_maximum": args.nodes_per_environment,
        "probe_count_per_run": len(probes),
        "curriculum_manifest": manifest,
        "runs": runs,
        "aggregate": {
            "run_count": len(runs),
            "probe_count": sum(run["probe_count"] for run in runs),
            "runs_with_any_local_argmax_change": sum(run["local_argmax_change_count"] > 0 for run in runs),
            "runs_with_any_projected_node_change": sum(run["projected_node_change_count"] > 0 for run in runs),
            "runs_with_any_projected_vector_change": sum(run["projected_vector_change_count"] > 0 for run in runs),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    for run in runs:
        print(
            run["run_name"],
            f"local={run['local_argmax_change_count']}/{run['probe_count']}",
            f"node={run['projected_node_change_count']}/{run['probe_count']}",
            f"joint={run['projected_vector_change_count']}/{run['probe_count']}",
            f"relative_q_median={run['relative_q_difference']['median']:.6g}",
        )


if __name__ == "__main__":
    main()
