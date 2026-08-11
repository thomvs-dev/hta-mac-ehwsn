"""Parallel no-learning sweep of a CH-safe cumulative QoS-deficit override."""

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
from experiments.diagnose_step3_delivery_feasibility import aggregate, build_environments, load_agent, sha256
from experiments.sweep_step3_risk_gated_completion import forwarding_energy_j


def qos_deficit_override(
    base_action, q_values, caps, mask, env, *, trajectory_target,
    reserve_floor, completion_fraction,
):
    """Override Q only to reduce the registered cumulative delivery deficit."""
    action = np.asarray(base_action, dtype=np.int64).copy()
    caps = np.asarray(caps, dtype=np.int64)
    mask = np.asarray(mask, dtype=bool)
    ch = int(env.ch)
    budget = int(env.base.cfg.frame_slot_budget)
    offered = int(env.base.queue[env.members][env.base.alive[env.members]].sum())
    counts = env.step3_qos_counts
    predicted_demand = int(counts["demand"]) + offered
    required_delivery = int(math.ceil(float(trajectory_target) * predicted_demand))
    predicted_base_delivery = int(counts["delivered"]) + int(action.sum())
    deficit_after_base = max(0, required_delivery - predicted_base_delivery)
    requested_additions = int(math.ceil(float(completion_fraction) * deficit_after_base))
    opportunity = max(0, min(budget, int(caps.sum())) - int(action.sum()))
    addition_cap = min(requested_additions, opportunity)
    audit = {
        "triggered": int(deficit_after_base > 0),
        "deficit_after_base": deficit_after_base,
        "requested_additions": requested_additions,
        "opportunity": opportunity,
        "added": 0,
        "risk_blocked": 0,
    }
    if addition_cap <= 0 or not env.base.alive[ch]:
        return action, audit
    added = 0
    while added < addition_cap and int(action.sum()) < budget:
        candidates = []
        for node in np.flatnonzero(mask & (action < caps)):
            level = int(action[node])
            gain = float(q_values[node, level + 1] - q_values[node, level])
            candidates.append((gain, -int(node), int(node)))
        if not candidates:
            break
        _, _, node = max(candidates)
        proposed_packets = int(action.sum()) + 1
        post_energy = float(env.base.energy[ch]) - forwarding_energy_j(env, ch, proposed_packets)
        if post_energy < float(reserve_floor) * env.base.cfg.initial_energy_j:
            audit["risk_blocked"] += addition_cap - added
            break
        action[node] += 1
        added += 1
    audit["added"] = added
    return action, audit


def _worker(paths, candidate):
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    threads = int(paths[6])
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, min(2, threads)))
    except RuntimeError:
        pass
    checkpoint, profile, risk_path, qos_path = map(Path, paths[:4])
    horizon = int(paths[4])
    agent, _ = load_agent(checkpoint)
    risk = json.loads(risk_path.read_text())
    qos = Step3QoSConstraintConfig.from_payload(json.loads(qos_path.read_text()))
    environments, _, cfg = build_environments(profile, risk, horizon)
    target, floor, fraction = candidate
    rows = []
    override_totals = {
        "triggered_steps": 0, "cumulative_deficit_after_base": 0,
        "requested_additions": 0, "available_opportunity": 0,
        "added_slots": 0, "risk_blocked_slots": 0,
    }
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
            members, ch = env.members.copy(), int(env.ch)
            alive_members = env.base.alive[members]
            ch_alive = bool(env.base.alive[ch])
            padded, padded_mask, caps = trainer.padded_state(env, observation, mask, env.base.n_nodes)
            base_action, q_values = agent.act(
                padded, padded_mask, epsilon=0.0, caps=caps,
                budget=cfg.frame_slot_budget,
                tie_break_priorities=np.arange(env.base.n_nodes),
            )
            action, audit = qos_deficit_override(
                base_action, q_values, caps, padded_mask, env,
                trajectory_target=target, reserve_floor=floor,
                completion_fraction=fraction,
            )
            offered = int(env.base.queue[members][alive_members].sum()) if ch_alive else 0
            feasible = int(caps.sum()) if ch_alive else 0
            oracle_delivery = min(feasible, int(cfg.frame_slot_budget))
            next_observation, next_mask, done, info = env.step(action)
            executed = int(info["target_packets_delivered"])
            role = info["energy_trace"]["role_energy"]
            next_members, next_ch = env.members.copy(), int(env.ch)
            transition = not np.array_equal(members, next_members) or ch != next_ch
            gap = max(0, oracle_delivery - executed)
            counters["offered_demand"] += offered
            counters["feasible_backlog"] += feasible
            counters["budget_feasible_delivery"] += oracle_delivery
            counters["requested_slots"] += int(base_action.sum())
            counters["requested_service"] += int(base_action.sum())
            counters["projected_slots"] += int(action.sum())
            counters["executed_delivery"] += executed
            counters["unused_budget"] += max(0, int(cfg.frame_slot_budget) - int(action.sum()))
            counters["service_gap_to_budget_oracle"] += gap
            counters["stale_drops"] += int(info["target_stale_drops"])
            counters["empty_target_steps"] += int(len(members) == 0)
            counters["dead_ch_before_steps"] += int(not ch_alive)
            counters["ch_death_terminations"] += int(ch_alive and not env.base.alive[ch])
            counters["membership_transitions"] += int(not np.array_equal(members, next_members))
            counters["ch_transitions"] += int(ch != next_ch)
            counters["post_transition_service_gap"] += int(previous_transition) * gap
            counters["member_tx_energy_j"] += float(role["member_tx"][members].sum())
            counters["ch_forwarding_energy_j"] += (
                float(role["ch_rx"][ch]) + float(role["ch_aggregate"][ch])
                + float(role["ch_tx_bs"][ch])
            )
            override_totals["triggered_steps"] += audit["triggered"]
            override_totals["cumulative_deficit_after_base"] += audit["deficit_after_base"]
            override_totals["requested_additions"] += audit["requested_additions"]
            override_totals["available_opportunity"] += audit["opportunity"]
            override_totals["added_slots"] += audit["added"]
            override_totals["risk_blocked_slots"] += audit["risk_blocked"]
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
    return {
        "candidate_id": f"target{target:.3f}_floor{floor:.2f}_fill{fraction:.2f}",
        "delivery_trajectory_target": target,
        "reserve_floor": floor,
        "completion_fraction": fraction,
        "aggregate": aggregate(rows, qos),
        "override": override_totals,
        "pairs": rows,
        "worker_pid": os.getpid(),
        "worker_wall_seconds": time.perf_counter() - wall_start,
        "worker_cpu_seconds": time.process_time() - cpu_start,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--environment-profile", type=Path, required=True)
    parser.add_argument("--ch-risk-config", type=Path, required=True)
    parser.add_argument("--qos-config", type=Path, required=True)
    parser.add_argument("--baseline-diagnostic", type=Path, required=True)
    parser.add_argument("--sweep-contract", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-worker", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resolve = lambda path: path if path.is_absolute() else ROOT / path
    contract_path = resolve(args.sweep_contract)
    contract = json.loads(contract_path.read_text())
    if contract.get("status") != "frozen_before_execution":
        raise RuntimeError("QoS-deficit sweep contract is not frozen")
    grid = contract["grid"]
    candidates = list(itertools.product(
        grid["delivery_trajectory_target"],
        grid["ch_post_forwarding_reserve_floor"],
        grid["deficit_completion_fraction"],
    ))
    logical = os.cpu_count() or 1
    if args.workers * args.threads_per_worker > logical:
        raise ValueError("configured worker threads exceed logical processors")
    baseline_path = resolve(args.baseline_diagnostic)
    baseline = json.loads(baseline_path.read_text())
    baseline_fnd = float(baseline["results"]["trained_greedy"]["aggregate"]["mean_fnd_free_steps"])
    paths = (
        str(resolve(args.checkpoint)), str(resolve(args.environment_profile)),
        str(resolve(args.ch_risk_config)), str(resolve(args.qos_config)),
        int(contract["horizon"]), str(contract_path), args.threads_per_worker,
    )
    wall_start = time.perf_counter()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_worker, paths, candidate) for candidate in candidates]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row = future.result()
            results.append(row)
            print(
                f"CANDIDATE={index}/{len(candidates)} ID={row['candidate_id']} "
                f"QOS={row['aggregate']['joint_qos_pass_count']}/20 "
                f"FND={row['aggregate']['mean_fnd_free_steps']:.2f}", flush=True,
            )
    wall = time.perf_counter() - wall_start
    required = int(contract["gates"]["minimum_joint_qos_pairs"])
    margin = float(contract["gates"]["fnd_noninferiority_margin_rounds"])
    for row in results:
        row["gates"] = {
            "joint_qos": row["aggregate"]["joint_qos_pass_count"] >= required,
            "fnd_noninferior": row["aggregate"]["mean_fnd_free_steps"] >= baseline_fnd - margin,
        }
        row["overall_pass"] = all(row["gates"].values())
    passing = [row for row in results if row["overall_pass"]]
    passing.sort(key=lambda row: (
        -row["aggregate"]["mean_fnd_free_steps"],
        row["override"]["added_slots"],
        -row["aggregate"]["macro_mean_delivery_ratio"],
        row["candidate_id"],
    ))
    selected = passing[0]["candidate_id"] if passing else None
    useful_cpu = sum(row["worker_cpu_seconds"] for row in results)
    output = resolve(args.output)
    payload = {
        "schema_version": 1,
        "status": "qos_deficit_candidate_selected" if selected else "no_qos_deficit_candidate_passed_stop_training",
        "development_only": True, "no_learning": True,
        "workers": args.workers,
        "threads_per_worker": args.threads_per_worker,
        "configured_compute_threads": args.workers * args.threads_per_worker,
        "logical_processors": logical,
        "wall_seconds": wall,
        "summed_worker_cpu_seconds": useful_cpu,
        "useful_worker_cpu_percent_of_logical_capacity": 100.0 * useful_cpu / (wall * logical),
        "sweep_contract_sha256": sha256(contract_path),
        "baseline_diagnostic_sha256": sha256(baseline_path),
        "selected_candidate": selected,
        "passing_candidates": [row["candidate_id"] for row in passing],
        "candidates": sorted(results, key=lambda row: row["candidate_id"]),
        "next_bounded_training_authorized": bool(selected),
        "claim_boundary": contract["claim_boundary"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": payload["status"], "selected_candidate": selected,
        "passing_candidates": payload["passing_candidates"],
        "wall_seconds": wall,
        "useful_worker_cpu_percent_of_logical_capacity": payload["useful_worker_cpu_percent_of_logical_capacity"],
        "output": str(output),
    }, indent=2))
    raise SystemExit(0 if selected else 3)


if __name__ == "__main__":
    main()
