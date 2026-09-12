"""Two-sided primary shield-intervention distillation sweep."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
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
from agents.ch_depletion_risk import validate_ch_risk_config
from agents.qos_constraints_v3 import Step3QoSConstraintConfig
from experiments.diagnose_step3_delivery_feasibility import build_environments, load_agent
from experiments.distill_step3_qos_shield import agreement, set_cpu_contract, train_distillation
from experiments.sweep_step3_qos_band_projection import qos_band_projection


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha256(path: str | Path) -> str:
    return hashlib.sha256(resolve(path).read_bytes()).hexdigest()


def load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text())
    if contract.get("status") != "frozen_before_primary_intervention_sweep":
        raise RuntimeError("intervention sweep contract is not frozen")
    for field in ("source_checkpoint", "oracle_report", "qos_config", "risk_config"):
        if sha256(contract[field]) != contract[f"{field}_sha256"]:
            raise RuntimeError(f"artifact hash mismatch: {field}")
    if set(contract["development_seeds"]).intersection(contract["prohibited_seeds"]):
        raise RuntimeError("prohibited seed requested")
    return contract


def project(agent, env, observation, mask, contract):
    padded, padded_mask, caps = trainer.padded_state(env, observation, mask, env.base.n_nodes)
    base_action, q_values = agent.act(
        padded, padded_mask, epsilon=0.0, caps=caps, budget=int(contract["budget"]),
        tie_break_priorities=np.arange(env.base.n_nodes),
    )
    band = contract["band"]
    demonstrated, audit = qos_band_projection(
        base_action, q_values, caps, padded_mask, env,
        lower_target=float(band["lower_delivery_target"]),
        upper_target=float(band["upper_delivery_target"]),
        reserve_floor=float(band["reserve_floor"]),
        completion_fraction=float(band["completion_fraction"]),
    )
    return padded, padded_mask, caps, base_action, demonstrated, audit


def collect(agent, environments, contract):
    states, actions, masks, caps_rows = [], [], [], []
    totals = {"steps": 0, "intervention_steps": 0, "added_slots": 0, "removed_slots": 0, "risk_blocked_slots": 0, "changed_slots_l1": 0}
    for env in environments:
        observation, mask, _ = env.reset()
        done = False
        while not done:
            padded, padded_mask, caps, base, demonstrated, audit = project(agent, env, observation, mask, contract)
            difference = int(np.abs(base - demonstrated).sum())
            if difference:
                states.append(np.asarray(padded, dtype=np.float16))
                actions.append(np.asarray(demonstrated, dtype=np.uint8))
                masks.append(np.asarray(padded_mask, dtype=bool))
                caps_rows.append(np.asarray(caps, dtype=np.uint8))
                totals["intervention_steps"] += 1
            totals["steps"] += 1
            totals["added_slots"] += int(audit["added"])
            totals["removed_slots"] += int(audit["removed"])
            totals["risk_blocked_slots"] += int(audit["risk_blocked"])
            totals["changed_slots_l1"] += difference
            observation, mask, done, _ = env.step(demonstrated)
    if not states:
        raise RuntimeError("two-sided primary shield produced no demonstrations")
    return (np.stack(states), np.stack(actions), np.stack(masks), np.stack(caps_rows)), totals


def evaluate(agent, environments, qos, contract):
    rows = []
    totals = {"steps": 0, "intervention_steps": 0, "added_slots": 0, "removed_slots": 0, "risk_blocked_slots": 0, "changed_slots_l1": 0}
    for env in environments:
        observation, mask, _ = env.reset()
        energy = 0.0
        done = False
        while not done:
            _, _, _, base, action, audit = project(agent, env, observation, mask, contract)
            difference = int(np.abs(base - action).sum())
            totals["steps"] += 1
            totals["intervention_steps"] += int(difference > 0)
            totals["added_slots"] += int(audit["added"])
            totals["removed_slots"] += int(audit["removed"])
            totals["risk_blocked_slots"] += int(audit["risk_blocked"])
            totals["changed_slots_l1"] += difference
            observation, mask, done, info = env.step(action)
            energy += float(np.asarray(info["energy_trace"]["consumed"]).sum())
        counts = env.step3_qos_counts
        demand = max(1, int(counts["demand"]))
        delivery = int(counts["delivered"]) / demand
        stale = int(counts["stale"]) / demand
        fairness = float(counts["episode_service_fairness"])
        rows.append({
            "seed": int(env.seed), "target_rank": int(env.target_rank),
            "delivery_ratio": delivery, "stale_ratio": stale,
            "episode_service_fairness": fairness,
            "joint_qos_pass": bool(delivery >= qos.minimum_delivery_ratio and stale <= qos.maximum_stale_drop_ratio and fairness >= qos.minimum_queue_fairness),
            "fnd_free_steps": int(env.base.t_fnd if env.base.t_fnd is not None else contract["horizon"]),
            "global_packets": int(env.base.total_packets),
            "network_energy_j": energy,
            "global_packets_per_j": int(env.base.total_packets) / max(energy, 1e-12),
        })
    return {
        "pairs": len(rows),
        "joint_qos_pass_count": int(sum(row["joint_qos_pass"] for row in rows)),
        "mean_delivery_ratio": float(np.mean([row["delivery_ratio"] for row in rows])),
        "mean_stale_ratio": float(np.mean([row["stale_ratio"] for row in rows])),
        "mean_episode_service_fairness": float(np.mean([row["episode_service_fairness"] for row in rows])),
        "mean_fnd_free_steps": float(np.mean([row["fnd_free_steps"] for row in rows])),
        "mean_global_packets_per_j": float(np.mean([row["global_packets_per_j"] for row in rows])),
        "intervention": totals,
        "rows": rows,
    }


def candidate_worker(contract_path_string, dataset_path_string, candidate, baseline):
    contract = load_contract(resolve(contract_path_string))
    set_cpu_contract(int(contract["threads_per_candidate"]), int(candidate["seed"]))
    data = np.load(dataset_path_string)
    dataset = tuple(data[name] for name in ("states", "actions", "masks", "caps"))
    agent, _ = load_agent(resolve(contract["source_checkpoint"]))
    history, train_idx, validation_idx = train_distillation(
        agent, dataset, epochs=int(candidate["epochs"]), batch_size=64,
        learning_rate=float(candidate["learning_rate"]), margin=float(candidate["margin"]),
        seed=int(candidate["seed"]),
    )
    validation = agreement(agent, *(array[validation_idx] for array in dataset), int(contract["budget"]), 64)
    risk = validate_ch_risk_config(json.loads(resolve(contract["risk_config"]).read_text()))
    qos = Step3QoSConstraintConfig.from_payload(json.loads(resolve(contract["qos_config"]).read_text()))
    envs, _, _ = build_environments(None, risk, int(contract["horizon"]), seeds=contract["development_seeds"])
    result = evaluate(agent, envs, qos, contract)
    gates = contract["gates"]
    reduction = 1.0 - result["intervention"]["changed_slots_l1"] / max(1, baseline["intervention"]["changed_slots_l1"])
    checks = {
        "joint_qos": result["joint_qos_pass_count"] >= int(gates["minimum_joint_qos_pairs"]),
        "intervention_reduction": reduction >= float(gates["minimum_intervention_slot_reduction_fraction"]),
        "fnd": result["mean_fnd_free_steps"] >= baseline["mean_fnd_free_steps"] - float(gates["maximum_mean_fnd_degradation_rounds"]),
        "fairness": result["mean_episode_service_fairness"] >= baseline["mean_episode_service_fairness"] - float(gates["maximum_fairness_degradation"]),
        "packets_per_j": result["mean_global_packets_per_j"] >= baseline["mean_global_packets_per_j"] * (1.0 - float(gates["maximum_packets_per_j_degradation_fraction"])),
    }
    output_dir = ROOT / "outputs" / "phase2" / "step3_primary_intervention_distillation_sweep_v1" / candidate["candidate_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "branching_c51_intervention_distilled.pt"
    agent.save(checkpoint, {"method": "primary_two_sided_shield_intervention_distillation", "candidate": candidate})
    return {
        "candidate": candidate, "history": history, "validation_agreement": validation,
        "evaluation": result, "intervention_reduction_fraction": reduction,
        "checks": checks, "gate_pass": all(checks.values()),
        "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract_path, output_path = resolve(args.contract), resolve(args.output)
    contract = load_contract(contract_path)
    started = time.perf_counter()
    risk = validate_ch_risk_config(json.loads(resolve(contract["risk_config"]).read_text()))
    qos = Step3QoSConstraintConfig.from_payload(json.loads(resolve(contract["qos_config"]).read_text()))
    source, _ = load_agent(resolve(contract["source_checkpoint"]))
    envs, _, _ = build_environments(None, risk, int(contract["horizon"]), seeds=contract["development_seeds"])
    dataset, collection = collect(source, envs, contract)
    dataset_path = ROOT / "outputs" / "phase2" / "step3_primary_intervention_distillation_sweep_v1" / "demonstrations.npz"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dataset_path, states=dataset[0], actions=dataset[1], masks=dataset[2], caps=dataset[3])
    baseline_envs, _, _ = build_environments(None, risk, int(contract["horizon"]), seeds=contract["development_seeds"])
    baseline = evaluate(source, baseline_envs, qos, contract)
    candidates = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(contract["parallel_candidates"])) as pool:
        futures = [pool.submit(candidate_worker, str(contract_path), str(dataset_path), item, baseline) for item in contract["candidates"]]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            candidates.append(row)
            print(f"CANDIDATE_COMPLETE={row['candidate']['candidate_id']} PASS={row['gate_pass']}", flush=True)
    order = {item["candidate_id"]: index for index, item in enumerate(contract["candidates"])}
    candidates.sort(key=lambda row: order[row["candidate"]["candidate_id"]])
    passing = [row for row in candidates if row["gate_pass"]]
    selected = min(passing, key=lambda row: (
        row["evaluation"]["intervention"]["changed_slots_l1"],
        -row["evaluation"]["mean_episode_service_fairness"],
        -row["evaluation"]["mean_global_packets_per_j"],
        -row["evaluation"]["mean_fnd_free_steps"],
        order[row["candidate"]["candidate_id"]],
    )) if passing else None
    payload = {
        "schema_version": 1,
        "status": "primary_intervention_candidate_selected" if selected else "no_primary_intervention_candidate_passed",
        "contract_sha256": sha256(contract_path), "evaluator_sha256": sha256(Path(__file__)),
        "collection": collection, "demonstrations": len(dataset[0]),
        "baseline": baseline, "candidates": candidates,
        "selected_candidate_id": selected["candidate"]["candidate_id"] if selected else None,
        "selected_checkpoint": selected["checkpoint"] if selected else None,
        "selected_checkpoint_sha256": selected["checkpoint_sha256"] if selected else None,
        "longer_training_authorized": bool(selected),
        "confirmation_seeds_opened": False,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": contract["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "selected": payload["selected_candidate_id"]}, indent=2), flush=True)
    print(f"OUTPUT={output_path}", flush=True)
    return 0 if selected else 3


if __name__ == "__main__":
    raise SystemExit(main())
