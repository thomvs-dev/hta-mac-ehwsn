"""Exact reduced-neighborhood oracle gap audit for B4 MAC allocations."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.train_phase2_dynamic_curriculum as trainer
from agents.guarded_marginal_teacher import GuardedTeacherConfig, guarded_marginal_action
from core.energy.idle_model import idle_listening_energy
from envs.step3_policy_observation import STEP3_CH_CONTEXT_SCHEMA
from experiments.distill_step3_qos_shield import set_cpu_contract
from experiments.evaluate_step4_publication_evidence import build_transfer_environments


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(value: str | Path) -> str:
    return hashlib.sha256(resolve(value).read_bytes()).hexdigest()


def load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("status") != "frozen_before_v2_phase_b5_reduced_cluster_exact_oracle":
        raise RuntimeError("B5 oracle contract is not frozen")
    if contract.get("phase") != "B5_oracle_gap_audit_no_training":
        raise RuntimeError("only the no-training oracle audit is authorized")
    for field in ("source_b4_stop_evidence", "qos_config", "risk_config"):
        if sha256(contract[field]) != contract[f"{field}_sha256"]:
            raise RuntimeError(f"artifact checksum mismatch: {field}")
    source = json.loads(resolve(contract["source_b4_stop_evidence"]).read_text(encoding="utf-8"))
    if source.get("status") != "phase_b4_failed_stop" or source.get("neural_training_performed") is not False:
        raise RuntimeError("B5 requires preserved failed B4 evidence")
    seeds = set(contract["development_seeds"])
    if seeds & set(contract["forbidden_confirmation_seeds"]):
        raise RuntimeError("forbidden confirmation seed requested")
    if int(contract["reduced_member_limit"]) > 8:
        raise RuntimeError("reduced oracle is capped at eight members")
    if int(contract["workers"]) * int(contract["threads_per_worker"]) > 8:
        raise RuntimeError("worker contract exceeds stable host capacity")
    return contract


def _member_tx(env) -> np.ndarray:
    values = np.full(env.base.n_nodes, np.inf, dtype=np.float64)
    for node in env.members:
        values[node] = float(env.base.radio.tx(
            env.base.cfg.packet_bits,
            float(np.linalg.norm(env.base.positions[node] - env.base.positions[int(env.ch)])),
        ))
    return values


def _served_expiring_by_count(env, node: int, count: int) -> int:
    return sum(int(age >= env.base.cfg.packet_ttl_rounds) for age in env.base.packet_ages[node][:int(count)])


def _choose_reduced_members(env, action, caps, mask, costs, limit: int) -> np.ndarray:
    served = [int(node) for node in np.flatnonzero(action > 0)]
    donors = sorted(served, key=lambda node: (-float(costs[node]), tuple(env.base.positions[node])))
    alternatives = sorted(
        (int(node) for node in np.flatnonzero(mask & (caps > action))),
        key=lambda node: (
            -_served_expiring_by_count(env, node, int(caps[node])),
            float(costs[node]), tuple(env.base.positions[node]),
        ),
    )
    donor_limit = min(len(donors), max(1, limit // 2))
    chosen = donors[:donor_limit]
    for node in alternatives + donors[donor_limit:]:
        if node not in chosen:
            chosen.append(node)
        if len(chosen) == limit:
            break
    return np.asarray(chosen, dtype=np.int64)


def exact_reduced_oracle(env, base_action, caps, mask, limit: int) -> tuple[np.ndarray, dict]:
    """Enumerate every allocation in a reduced member neighborhood exactly."""
    base_action = np.asarray(base_action, dtype=np.int64)
    caps = np.asarray(caps, dtype=np.int64); mask = np.asarray(mask, dtype=bool)
    tx = _member_tx(env)
    local = _choose_reduced_members(env, base_action, caps, mask, tx, limit)
    if len(local) < 2:
        return base_action.copy(), {"enumerated": 1, "feasible": 1, "local_members": local.tolist()}
    local_slots = int(base_action[local].sum())
    frame_slots = int(base_action.sum())
    ranges = [range(int(caps[node]) + 1) for node in local]
    combos = np.asarray(list(itertools.product(*ranges)), dtype=np.int16)
    combos = combos[combos.sum(axis=1) == local_slots]
    if not len(combos):
        raise RuntimeError("base allocation disappeared from exact reduced action set")

    outside = base_action.copy(); outside[local] = 0
    baseline_expiring = sum(_served_expiring_by_count(env, int(node), int(base_action[node])) for node in np.flatnonzero(base_action))
    outside_expiring = sum(_served_expiring_by_count(env, int(node), int(outside[node])) for node in np.flatnonzero(outside))
    stale_tables = [
        [_served_expiring_by_count(env, int(node), count) for count in range(int(caps[node]) + 1)]
        for node in local
    ]
    local_expiring = np.zeros(len(combos), dtype=np.int16)
    for column in range(len(local)):
        local_expiring += np.take(stale_tables[column], combos[:, column])
    stale_ok = outside_expiring + local_expiring >= baseline_expiring

    offered = env.step3_episode_offered_per_node.astype(np.float64) + env.base.queue
    delivered_outside = env.step3_episode_delivered_per_node.astype(np.float64) + outside
    eligible = offered > 0.0
    outside_nodes = eligible.copy(); outside_nodes[local] = False
    outside_ratios = np.clip(delivered_outside[outside_nodes] / offered[outside_nodes], 0.0, 1.0)
    ratio_sum = float(outside_ratios.sum()); ratio_square_sum = float(np.square(outside_ratios).sum())
    local_ratios = np.clip(
        (env.step3_episode_delivered_per_node[local][None, :] + combos) / offered[local][None, :],
        0.0, 1.0,
    )
    sums = ratio_sum + local_ratios.sum(axis=1)
    squares = ratio_square_sum + np.square(local_ratios).sum(axis=1)
    cohort = int(eligible.sum())
    fairness = np.divide(np.square(sums), cohort * squares, out=np.zeros_like(sums), where=squares > 0.0)
    base_delivered = env.step3_episode_delivered_per_node.astype(np.float64) + base_action
    base_ratios = np.clip(base_delivered[eligible] / offered[eligible], 0.0, 1.0)
    base_fairness = float(base_ratios.sum() ** 2 / (cohort * np.square(base_ratios).sum()))
    fairness_ok = fairness + 1e-12 >= base_fairness

    post_local = env.base.energy[local][None, :] - combos * tx[local][None, :]
    outside_members = np.asarray([node for node in env.members if node not in set(local.tolist())], dtype=np.int64)
    outside_min = float(np.min(env.base.energy[outside_members] - outside[outside_members] * tx[outside_members])) if len(outside_members) else np.inf
    minimum_post = np.minimum(post_local.min(axis=1), outside_min)
    base_minimum_post = float(np.min(env.base.energy[env.members] - base_action[env.members] * tx[env.members]))
    residual_ok = minimum_post + 1e-15 >= base_minimum_post
    feasible = stale_ok & fairness_ok & residual_ok
    if not np.any(feasible):
        raise RuntimeError("base allocation failed its own reduced-oracle constraints")
    idle_per_slot = float(idle_listening_energy(
        1,
        p_idle_j_per_bit_time=env.base.cfg.e_elec_j_per_bit,
        slot_bit_times=env.base.cfg.idle_slot_bit_times,
    ))
    energy = np.zeros(len(combos), dtype=np.float64)
    for column, node in enumerate(local):
        counts = combos[:, column].astype(np.float64)
        energy += counts * tx[node]
        energy += np.where(counts > 0.0, (frame_slots - counts) * idle_per_slot, 0.0)
    indices = np.flatnonzero(feasible)
    winner = int(indices[np.argmin(energy[indices])])
    oracle = outside.copy(); oracle[local] = combos[winner]
    if np.any(oracle < 0) or np.any(oracle > caps) or np.any(oracle[~mask] != 0) or int(oracle.sum()) != int(base_action.sum()):
        raise RuntimeError("exact oracle produced an infeasible action")
    return oracle, {
        "enumerated": int(len(combos)), "feasible": int(feasible.sum()),
        "local_members": local.tolist(), "base_fairness": base_fairness,
        "oracle_fairness": float(fairness[winner]), "base_expiring": int(baseline_expiring),
        "oracle_expiring": int(outside_expiring + local_expiring[winner]),
        "base_minimum_post_energy": base_minimum_post,
        "oracle_minimum_post_energy": float(minimum_post[winner]),
        "base_member_tx_j": float(base_action[env.members] @ tx[env.members]),
        "oracle_member_tx_j": float(oracle[env.members] @ tx[env.members]),
        "base_reduced_tx_idle_j": float(
            sum(
                int(base_action[node]) * tx[node]
                + ((frame_slots - int(base_action[node])) * idle_per_slot if base_action[node] > 0 else 0.0)
                for node in local
            )
        ),
        "oracle_reduced_tx_idle_j": float(energy[winner]),
    }


def evaluate_seed(contract_path: str, seed: int) -> dict:
    contract = load_contract(resolve(contract_path))
    set_cpu_contract(int(contract["threads_per_worker"]), 127_000 + int(seed))
    runtime = dict(contract); runtime["development_seeds"] = [int(seed)]
    runtime["observation_schema"] = STEP3_CH_CONTEXT_SCHEMA
    envs = build_transfer_environments(runtime, contract["scenario"], trace=None)
    config = GuardedTeacherConfig.from_payload(contract["b4_candidate"])
    rows = []
    for env in envs:
        observation, mask, _ = env.reset(); done = False; round_index = 0; samples = 0
        while not done:
            _, active, caps = trainer.padded_state(env, observation, mask, env.base.n_nodes)
            base_action = guarded_marginal_action(env, active, caps, int(contract["budget"]), config)
            should_sample = (
                round_index % int(contract["sample_interval_rounds"]) == 0
                and samples < int(contract["maximum_samples_per_rank"])
                and int(base_action.sum()) > 0 and int(active.sum()) > 1
            )
            if should_sample:
                oracle_action, audit = exact_reduced_oracle(
                    env, base_action, caps, active, int(contract["reduced_member_limit"])
                )
                oracle_env = copy.deepcopy(env)
                _, _, _, oracle_info = oracle_env.step(oracle_action)
                _, _, _, base_info_copy = copy.deepcopy(env).step(base_action)
                base_energy = float(np.asarray(base_info_copy["energy_trace"]["consumed"]).sum())
                oracle_energy = float(np.asarray(oracle_info["energy_trace"]["consumed"]).sum())
                base_role = base_info_copy["energy_trace"]["role_energy"]
                oracle_role = oracle_info["energy_trace"]["role_energy"]
                role_names = ("member_tx", "ch_rx", "ch_aggregate", "ch_tx_bs", "idle")
                base_target_role = sum(float(np.asarray(base_role[name]).sum()) for name in role_names)
                oracle_target_role = sum(float(np.asarray(oracle_role[name]).sum()) for name in role_names)
                constraint_violations = int(
                    int(oracle_action.sum()) != int(base_action.sum())
                    or audit["oracle_expiring"] < audit["base_expiring"]
                    or audit["oracle_fairness"] + 1e-12 < audit["base_fairness"]
                    or audit["oracle_minimum_post_energy"] + 1e-15 < audit["base_minimum_post_energy"]
                )
                rows.append({
                    "seed": int(seed), "target_rank": int(env.target_rank), "round": round_index,
                    **audit, "action_changed": bool(np.any(oracle_action != base_action)),
                    "base_network_energy_j": base_energy, "oracle_network_energy_j": oracle_energy,
                    "base_target_role_energy_j": base_target_role,
                    "oracle_target_role_energy_j": oracle_target_role,
                    "target_role_energy_reduction_fraction": float((base_target_role - oracle_target_role) / max(base_target_role, 1e-12)),
                    "network_energy_reduction_fraction": float((base_energy - oracle_energy) / max(base_energy, 1e-12)),
                    "minimum_residual_gain_fraction": float((audit["oracle_minimum_post_energy"] - audit["base_minimum_post_energy"]) / max(audit["base_minimum_post_energy"], 1e-12)),
                    "constraint_violations": constraint_violations,
                })
                samples += 1
            observation, mask, done, _ = env.step(base_action)
            round_index += 1
    return {"seed": int(seed), "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--contract", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); contract_path, output_path = resolve(args.contract), resolve(args.output)
    contract = load_contract(contract_path); started = time.perf_counter(); tasks = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(contract["workers"])) as pool:
        futures = [pool.submit(evaluate_seed, str(contract_path), int(seed)) for seed in contract["development_seeds"]]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            tasks.append(future.result()); print(f"PHASE_B5_PROGRESS={index}/{len(futures)}", flush=True)
    rows = [row for task in tasks for row in task["rows"]]
    improved = [row for row in rows if row["target_role_energy_reduction_fraction"] > 1e-12]
    gate = contract["continuation_gate"]
    metrics = {
        "sampled_states": len(rows), "improved_states": len(improved),
        "improved_state_fraction": len(improved) / max(1, len(rows)),
        "mean_target_role_energy_reduction_fraction_on_improved_states": float(np.mean([row["target_role_energy_reduction_fraction"] for row in improved])) if improved else 0.0,
        "mean_network_energy_reduction_fraction_on_improved_states": float(np.mean([row["network_energy_reduction_fraction"] for row in improved])) if improved else 0.0,
        "mean_minimum_residual_gain_fraction_on_improved_states": float(np.mean([row["minimum_residual_gain_fraction"] for row in improved])) if improved else 0.0,
        "constraint_violations": int(sum(row["constraint_violations"] for row in rows)),
        "mean_enumerated_actions": float(np.mean([row["enumerated"] for row in rows])) if rows else 0.0,
        "mean_feasible_actions": float(np.mean([row["feasible"] for row in rows])) if rows else 0.0,
    }
    checks = {
        "opportunity_frequency": metrics["improved_state_fraction"] >= float(gate["minimum_improved_state_fraction"]),
        "energy_headroom": metrics["mean_target_role_energy_reduction_fraction_on_improved_states"] >= float(gate["minimum_mean_target_role_energy_reduction_fraction_on_improved_states"]),
        "residual_nondegradation": metrics["mean_minimum_residual_gain_fraction_on_improved_states"] >= float(gate["minimum_mean_minimum_residual_gain_fraction_on_improved_states"]),
        "constraints": metrics["constraint_violations"] <= int(gate["maximum_constraint_violations"]),
        "no_training": True,
    }
    passed = all(checks.values())
    payload = {
        "schema_version": 1, "status": "phase_b5_oracle_gap_pass" if passed else "phase_b5_oracle_gap_fail_stop",
        "contract_sha256": sha256(contract_path), "runner_sha256": sha256(Path(__file__)),
        "metrics": metrics, "checks": checks, "oracle_gap_pass": passed,
        "new_deterministic_teacher_authorized": passed, "neural_training_authorized": False,
        "neural_training_performed": False, "rows": sorted(rows, key=lambda row: (row["seed"], row["target_rank"], row["round"])),
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": "exact enumeration within sampled eight-member action neighborhoods; not a global finite-horizon oracle",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True); output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "metrics": metrics, "checks": checks, "output": str(output_path)}, indent=2), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
