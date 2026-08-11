"""Parallel no-learning sweep of risk-gated Step 3 budget completion."""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.train_phase2_dynamic_curriculum as trainer
from agents.qos_constraints_v3 import Step3QoSConstraintConfig
from experiments.diagnose_step3_delivery_feasibility import (
    aggregate,
    build_environments,
    load_agent,
    sha256,
)
from envs.fixed_cluster_training_env import FixedClusterTrainingEnv


def forwarding_energy_j(env, ch, packets):
    packets = int(packets)
    if packets <= 0:
        return 0.0
    bits = env.base.cfg.packet_bits * packets
    cost = env.base.radio.rx(bits) + env.base.radio.aggregate(bits)
    distance = float(np.linalg.norm(
        env.base.positions[int(ch)] - np.asarray(env.base.cfg.bs_position_m)
    ))
    return cost + env.base.radio.tx(env.base.cfg.packet_bits, distance)


def risk_gated_complete(
    base_action, q_values, caps, mask, env, *, reserve_floor,
    completion_fraction, negative_tolerance_factor,
):
    """Add selected feasible slots without violating a no-harvest CH floor."""
    action = np.asarray(base_action, dtype=np.int64).copy()
    caps = np.asarray(caps, dtype=np.int64)
    mask = np.asarray(mask, dtype=bool)
    ch = int(env.ch)
    budget = int(env.base.cfg.frame_slot_budget)
    opportunity = max(0, min(budget, int(caps.sum())) - int(action.sum()))
    addition_cap = int(math.ceil(float(completion_fraction) * opportunity))
    if addition_cap <= 0 or not env.base.alive[ch]:
        return action, {"added": 0, "risk_blocked": 0, "q_blocked": 0}

    marginal_values = []
    for node in np.flatnonzero(mask):
        for level in range(int(caps[node])):
            marginal_values.append(float(q_values[node, level + 1] - q_values[node, level]))
    q_scale = float(np.median(np.abs(marginal_values))) if marginal_values else 0.0
    tolerance = float(negative_tolerance_factor) * max(q_scale, 1e-12)
    added = 0
    risk_blocked = 0
    q_blocked = 0
    while added < addition_cap and int(action.sum()) < budget:
        candidates = []
        for node in np.flatnonzero(mask & (action < caps)):
            level = int(action[node])
            gain = float(q_values[node, level + 1] - q_values[node, level])
            candidates.append((gain, -int(node), int(node)))
        if not candidates:
            break
        gain, _, node = max(candidates)
        if gain < -tolerance:
            q_blocked += addition_cap - added
            break
        proposed_packets = int(action.sum()) + 1
        post_energy = float(env.base.energy[ch]) - forwarding_energy_j(
            env, ch, proposed_packets
        )
        minimum_energy = float(reserve_floor) * env.base.cfg.initial_energy_j
        if post_energy < minimum_energy:
            risk_blocked += addition_cap - added
            break
        action[node] += 1
        added += 1
    return action, {
        "added": added, "risk_blocked": risk_blocked,
        "q_blocked": q_blocked, "q_tolerance": tolerance,
    }


def _evaluate_candidate(paths, candidate):
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    threads_per_worker = int(paths[5])
    torch.set_num_threads(threads_per_worker)
    try:
        torch.set_num_interop_threads(max(1, min(2, threads_per_worker)))
    except RuntimeError:
        pass
    checkpoint = Path(paths[0])
    profile = Path(paths[1])
    risk_path = Path(paths[2])
    qos_path = Path(paths[3])
    horizon = int(paths[4])
    risk = json.loads(risk_path.read_text())
    qos = Step3QoSConstraintConfig.from_payload(json.loads(qos_path.read_text()))
    agent, _ = load_agent(checkpoint)
    environments, _, cfg = build_environments(profile, risk, horizon)
    reserve_floor, completion_fraction, tolerance_factor = candidate
    rows = []
    total_added = total_risk_blocked = total_q_blocked = 0
    for env in environments:
        observation, mask, _ = env.reset()
        counters = {
            "offered_demand": 0, "feasible_backlog": 0,
            "budget_feasible_delivery": 0, "requested_slots": 0,
            "requested_service": 0, "projected_slots": 0,
            "executed_delivery": 0, "unused_budget": 0,
            "service_gap_to_budget_oracle": 0, "stale_drops": 0,
            "empty_target_steps": 0, "dead_ch_before_steps": 0,
            "ch_death_terminations": 0, "membership_transitions": 0,
            "ch_transitions": 0, "post_transition_service_gap": 0,
            "member_tx_energy_j": 0.0, "ch_forwarding_energy_j": 0.0,
        }
        done = False
        steps = 0
        previous_transition = False
        while not done:
            members = env.members.copy()
            ch = int(env.ch)
            alive_members = env.base.alive[members]
            ch_alive = bool(env.base.alive[ch])
            padded, padded_mask, caps = trainer.padded_state(
                env, observation, mask, env.base.n_nodes
            )
            base_action, q_values = agent.act(
                padded, padded_mask, epsilon=0.0, caps=caps, budget=cfg.frame_slot_budget,
                tie_break_priorities=np.arange(env.base.n_nodes),
            )
            action, completion = risk_gated_complete(
                base_action, q_values, caps, padded_mask, env,
                reserve_floor=reserve_floor,
                completion_fraction=completion_fraction,
                negative_tolerance_factor=tolerance_factor,
            )
            offered = int(env.base.queue[members][alive_members].sum()) if ch_alive else 0
            feasible = int(caps.sum()) if ch_alive else 0
            oracle_delivery = min(feasible, int(cfg.frame_slot_budget))
            requested = int(base_action.sum())
            next_observation, next_mask, done, info = env.step(action)
            executed = int(info["target_packets_delivered"])
            role = info["energy_trace"]["role_energy"]
            next_members, next_ch = env.members.copy(), int(env.ch)
            transition = not np.array_equal(members, next_members) or ch != next_ch
            service_gap = max(0, oracle_delivery - executed)
            counters["offered_demand"] += offered
            counters["feasible_backlog"] += feasible
            counters["budget_feasible_delivery"] += oracle_delivery
            counters["requested_slots"] += requested
            counters["requested_service"] += requested
            counters["projected_slots"] += int(action.sum())
            counters["executed_delivery"] += executed
            counters["unused_budget"] += max(0, int(cfg.frame_slot_budget) - int(action.sum()))
            counters["service_gap_to_budget_oracle"] += service_gap
            counters["stale_drops"] += int(info["target_stale_drops"])
            counters["empty_target_steps"] += int(len(members) == 0)
            counters["dead_ch_before_steps"] += int(not ch_alive)
            counters["ch_death_terminations"] += int(ch_alive and not env.base.alive[ch])
            counters["membership_transitions"] += int(not np.array_equal(members, next_members))
            counters["ch_transitions"] += int(ch != next_ch)
            counters["post_transition_service_gap"] += int(previous_transition) * service_gap
            counters["member_tx_energy_j"] += float(role["member_tx"][members].sum())
            counters["ch_forwarding_energy_j"] += (
                float(role["ch_rx"][ch]) + float(role["ch_aggregate"][ch])
                + float(role["ch_tx_bs"][ch])
            )
            total_added += int(completion["added"])
            total_risk_blocked += int(completion["risk_blocked"])
            total_q_blocked += int(completion["q_blocked"])
            previous_transition = transition
            observation, mask = next_observation, next_mask
            steps += 1
        demand = max(1, counters["offered_demand"])
        delivery = counters["executed_delivery"] / demand
        stale = counters["stale_drops"] / demand
        fairness = float(env.step3_qos_counts["fairness"])
        rows.append({
            "seed": env.seed, "target_rank": env.target_rank, "steps": steps,
            **counters, "delivery_ratio": delivery, "stale_ratio": stale,
            "fairness": fairness,
            "joint_qos_pass": bool(
                delivery >= qos.minimum_delivery_ratio
                and stale <= qos.maximum_stale_drop_ratio
                and fairness >= qos.minimum_queue_fairness
            ),
            "fnd_free_steps": int(env.base.t_fnd if env.base.t_fnd is not None else steps),
        })
    result = {
        "candidate_id": f"floor{reserve_floor:.2f}_fill{completion_fraction:.2f}_tol{tolerance_factor:.2f}",
        "reserve_floor": reserve_floor,
        "completion_fraction": completion_fraction,
        "negative_tolerance_factor": tolerance_factor,
        "aggregate": aggregate(rows, qos),
        "completion": {
            "added_slots": total_added,
            "risk_blocked_slots": total_risk_blocked,
            "q_blocked_slots": total_q_blocked,
        },
        "pairs": rows,
        "worker_pid": os.getpid(),
        "worker_wall_seconds": time.perf_counter() - started_wall,
        "worker_cpu_seconds": time.process_time() - started_cpu,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--environment-profile", type=Path, required=True)
    parser.add_argument("--ch-risk-config", type=Path, required=True)
    parser.add_argument("--qos-config", type=Path, required=True)
    parser.add_argument("--baseline-diagnostic", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.horizon != 1200:
        raise ValueError("frozen sweep requires horizon 1200")
    if args.threads_per_worker < 1:
        raise ValueError("threads-per-worker must be positive")
    resolve = lambda path: path if path.is_absolute() else ROOT / path
    paths = (
        str(resolve(args.checkpoint)), str(resolve(args.environment_profile)),
        str(resolve(args.ch_risk_config)), str(resolve(args.qos_config)),
        args.horizon, args.threads_per_worker,
    )
    baseline_path = resolve(args.baseline_diagnostic)
    baseline = json.loads(baseline_path.read_text())
    baseline_fnd = float(baseline["results"]["trained_greedy"]["aggregate"]["mean_fnd_free_steps"])
    candidates = list(itertools.product(
        (0.10, 0.15, 0.20), (0.25, 0.50, 0.75), (0.00, 0.05)
    ))
    workers = min(max(1, args.workers), len(candidates), os.cpu_count() or 1)
    if workers * args.threads_per_worker > (os.cpu_count() or 1):
        raise ValueError("configured worker threads exceed logical processors")
    started = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_evaluate_candidate, paths, candidate) for candidate in candidates]
        results = []
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(
                f"CANDIDATE={index}/{len(candidates)} ID={result['candidate_id']} "
                f"QOS={result['aggregate']['joint_qos_pass_count']}/20 "
                f"FND={result['aggregate']['mean_fnd_free_steps']:.2f}",
                flush=True,
            )
    wall = time.perf_counter() - started
    useful_cpu = sum(row["worker_cpu_seconds"] for row in results)
    for row in results:
        agg = row["aggregate"]
        row["gates"] = {
            "joint_qos_18_of_20": agg["joint_qos_pass_count"] >= 18,
            "fnd_noninferior_12_rounds": agg["mean_fnd_free_steps"] >= baseline_fnd - 12.0,
        }
        row["overall_pass"] = all(row["gates"].values())
    passing = [row for row in results if row["overall_pass"]]
    passing.sort(key=lambda row: (
        -row["aggregate"]["mean_fnd_free_steps"],
        -row["aggregate"]["joint_qos_pass_count"],
        -row["aggregate"]["macro_mean_delivery_ratio"],
        row["candidate_id"],
    ))
    selected = passing[0]["candidate_id"] if passing else None
    output_path = resolve(args.output)
    payload = {
        "schema_version": 1,
        "status": "completion_candidate_selected" if selected else "no_completion_candidate_passed_stop_training",
        "development_only": True,
        "no_learning": True,
        "workers": workers,
        "threads_per_worker": args.threads_per_worker,
        "configured_compute_threads": workers * args.threads_per_worker,
        "logical_processors": os.cpu_count(),
        "wall_seconds": wall,
        "summed_worker_cpu_seconds": useful_cpu,
        "useful_worker_cpu_percent_of_logical_capacity": 100.0 * useful_cpu / max(1e-12, wall * (os.cpu_count() or 1)),
        "grid": {
            "reserve_floor": [0.10, 0.15, 0.20],
            "completion_fraction": [0.25, 0.50, 0.75],
            "negative_tolerance_factor": [0.00, 0.05],
        },
        "gates": {
            "joint_qos_passes_required": 18,
            "baseline_mean_fnd_free_steps": baseline_fnd,
            "fnd_noninferiority_margin_rounds": 12.0,
            "minimum_mean_fnd_free_steps": baseline_fnd - 12.0,
        },
        "selected_candidate": selected,
        "passing_candidates": [row["candidate_id"] for row in passing],
        "candidates": sorted(results, key=lambda row: row["candidate_id"]),
        "evidence": {
            "checkpoint_sha256": sha256(Path(paths[0])),
            "baseline_diagnostic_sha256": sha256(baseline_path),
        },
        "next_bounded_training_authorized": bool(selected),
        "claim_boundary": "parallel_development_no_learning_sweep_not_model_selection_or_publication_evidence",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": payload["status"], "selected_candidate": selected,
        "passing_candidates": payload["passing_candidates"],
        "wall_seconds": wall,
        "useful_worker_cpu_percent_of_logical_capacity": payload["useful_worker_cpu_percent_of_logical_capacity"],
        "output": str(output_path),
    }, indent=2))
    raise SystemExit(0 if selected else 3)


if __name__ == "__main__":
    main()
