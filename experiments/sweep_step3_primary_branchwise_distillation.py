"""Bounded branch-wise distillation of primary QoS-band interventions."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.ch_depletion_risk import validate_ch_risk_config
from agents.qos_constraints_v3 import Step3QoSConstraintConfig
from experiments.diagnose_step3_delivery_feasibility import build_environments, load_agent
from experiments.distill_step3_qos_shield import set_cpu_contract
from experiments.sweep_step3_primary_intervention_distillation import evaluate, project, resolve


def sha256(path: str | Path) -> str:
    return hashlib.sha256(resolve(path).read_bytes()).hexdigest()


def load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text())
    if contract.get("status") != "frozen_before_primary_branchwise_sweep":
        raise RuntimeError("branch-wise sweep contract is not frozen")
    for field in ("source_checkpoint", "qos_config", "risk_config"):
        if sha256(contract[field]) != contract[f"{field}_sha256"]:
            raise RuntimeError(f"artifact hash mismatch: {field}")
    if set(contract["development_seeds"]).intersection(contract["prohibited_seeds"]):
        raise RuntimeError("prohibited seed requested")
    return contract


def collect(agent, environments, contract):
    states, source_actions, targets, masks, caps_rows = [], [], [], [], []
    totals = {"steps": 0, "intervention_steps": 0, "added_slots": 0,
              "removed_slots": 0, "changed_slots_l1": 0}
    for env in environments:
        observation, mask, _ = env.reset()
        done = False
        while not done:
            padded, padded_mask, caps, base, target, audit = project(
                agent, env, observation, mask, contract
            )
            difference = int(np.abs(base - target).sum())
            if difference:
                states.append(np.asarray(padded, dtype=np.float16))
                source_actions.append(np.asarray(base, dtype=np.uint8))
                targets.append(np.asarray(target, dtype=np.uint8))
                masks.append(np.asarray(padded_mask, dtype=bool))
                caps_rows.append(np.asarray(caps, dtype=np.uint8))
                totals["intervention_steps"] += 1
            totals["steps"] += 1
            totals["added_slots"] += int(audit["added"])
            totals["removed_slots"] += int(audit["removed"])
            totals["changed_slots_l1"] += difference
            observation, mask, done, _ = env.step(target)
    if not states:
        raise RuntimeError("two-sided shield produced no demonstrations")
    return (
        np.stack(states), np.stack(source_actions), np.stack(targets),
        np.stack(masks), np.stack(caps_rows), totals,
    )


def masked_action_log_policy(agent, states, masks, caps, temperature):
    q_values = agent.online.q_values(agent._transform_state_tensor(states), masks)
    levels = torch.arange(agent.cfg.actions, device=agent.device).view(1, 1, -1)
    feasible = levels <= caps.unsqueeze(-1)
    logits = (q_values / float(temperature)).masked_fill(~feasible, -1e9)
    return F.log_softmax(logits, dim=-1)


def branchwise_loss(
    agent, source_agent, states, source_actions, targets, masks, caps, *,
    temperature, changed_weight, preservation_weight,
):
    current_log = masked_action_log_policy(agent, states, masks, caps, temperature)
    with torch.no_grad():
        source_log = masked_action_log_policy(
            source_agent, states, masks, caps, temperature
        )
        source_probability = source_log.exp()
    target_nll = -current_log.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    additions = masks & (targets > source_actions)
    removals = masks & (targets < source_actions)
    changed_terms = []
    if additions.any():
        changed_terms.append(target_nll[additions].mean())
    if removals.any():
        changed_terms.append(target_nll[removals].mean())
    if not changed_terms:
        raise RuntimeError("batch contains no changed intervention branch")
    changed_loss = torch.stack(changed_terms).mean()
    unchanged = masks & (targets == source_actions)
    kl = (source_probability * (source_log - current_log)).sum(dim=-1)
    preservation_loss = kl[unchanged].mean() if unchanged.any() else kl.new_zeros(())
    total = float(changed_weight) * changed_loss + float(preservation_weight) * preservation_loss
    return total, changed_loss, preservation_loss


@torch.no_grad()
def branch_metrics(agent, dataset, indices, budget, batch_size=64):
    states_np, source_np, targets_np, masks_np, caps_np = dataset
    counts = {"samples": len(indices), "exact_actions": 0, "absolute_error": 0,
              "addition_total": 0, "addition_correct": 0,
              "removal_total": 0, "removal_correct": 0,
              "unchanged_total": 0, "unchanged_correct": 0}
    agent.online.eval()
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        states = torch.as_tensor(states_np[selected], dtype=torch.float32, device=agent.device)
        masks = torch.as_tensor(masks_np[selected], dtype=torch.bool, device=agent.device)
        q_values = agent.online.q_values(agent._transform_state_tensor(states), masks).cpu().numpy()
        for offset, values in enumerate(q_values):
            index = int(selected[offset])
            predicted = agent._project(values, masks_np[index], caps=caps_np[index])
            source, target, valid = source_np[index], targets_np[index], masks_np[index]
            counts["exact_actions"] += int(np.array_equal(predicted, target))
            counts["absolute_error"] += int(np.abs(predicted - target).sum())
            for name, selector in (
                ("addition", valid & (target > source)),
                ("removal", valid & (target < source)),
                ("unchanged", valid & (target == source)),
            ):
                counts[f"{name}_total"] += int(selector.sum())
                counts[f"{name}_correct"] += int((predicted[selector] == target[selector]).sum())
    branches = len(indices) * targets_np.shape[1]
    result = {
        "samples": len(indices),
        "exact_action_fraction": counts["exact_actions"] / max(1, len(indices)),
        "mean_branch_absolute_error": counts["absolute_error"] / max(1, branches),
    }
    for name in ("addition", "removal", "unchanged"):
        total = counts[f"{name}_total"]
        result[f"{name}_total"] = total
        result[f"{name}_agreement"] = counts[f"{name}_correct"] / max(1, total)
    return result


def train_branchwise(agent, dataset, candidate, objective):
    states_np, source_np, targets_np, masks_np, caps_np = dataset
    rng = np.random.default_rng(int(candidate["seed"]))
    order = rng.permutation(len(states_np))
    split = max(1, int(0.8 * len(order)))
    train_idx, validation_idx = order[:split], order[split:]
    if not len(validation_idx):
        validation_idx = train_idx[-1:]
    source_agent = copy.deepcopy(agent)
    source_agent.online.eval()
    for parameter in source_agent.online.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(
        agent.online.parameters(), lr=float(candidate["learning_rate"]),
        weight_decay=float(objective["weight_decay"]),
    )
    history = []
    batch_size = int(objective["batch_size"])
    for epoch in range(1, int(candidate["epochs"]) + 1):
        shuffled = rng.permutation(train_idx)
        losses, changed_losses, preservation_losses = [], [], []
        agent.online.train()
        for start in range(0, len(shuffled), batch_size):
            selected = shuffled[start:start + batch_size]
            tensors = [
                torch.as_tensor(states_np[selected], dtype=torch.float32, device=agent.device),
                torch.as_tensor(source_np[selected], dtype=torch.long, device=agent.device),
                torch.as_tensor(targets_np[selected], dtype=torch.long, device=agent.device),
                torch.as_tensor(masks_np[selected], dtype=torch.bool, device=agent.device),
                torch.as_tensor(caps_np[selected], dtype=torch.long, device=agent.device),
            ]
            loss, changed, preservation = branchwise_loss(
                agent, source_agent, *tensors,
                temperature=float(candidate["temperature"]),
                changed_weight=float(candidate["changed_weight"]),
                preservation_weight=float(candidate["preservation_weight"]),
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                agent.online.parameters(), float(objective["gradient_clip"])
            )
            optimizer.step()
            losses.append(float(loss.item()))
            changed_losses.append(float(changed.item()))
            preservation_losses.append(float(preservation.item()))
        validation = branch_metrics(agent, dataset, validation_idx, agent.cfg.budget, batch_size)
        row = {
            "epoch": epoch, "mean_loss": float(np.mean(losses)),
            "mean_changed_loss": float(np.mean(changed_losses)),
            "mean_preservation_loss": float(np.mean(preservation_losses)),
            "validation": validation,
        }
        history.append(row)
        print(
            f"BRANCH_EPOCH={epoch}/{candidate['epochs']} LOSS={row['mean_loss']:.6f} "
            f"ADD={validation['addition_agreement']:.4f} "
            f"REMOVE={validation['removal_agreement']:.4f} "
            f"KEEP={validation['unchanged_agreement']:.4f}", flush=True,
        )
    agent.target.load_state_dict(agent.online.state_dict())
    return history, train_idx, validation_idx


def candidate_worker(contract_path_string, dataset_path_string, candidate, baseline):
    contract = load_contract(resolve(contract_path_string))
    set_cpu_contract(int(contract["threads_per_candidate"]), int(candidate["seed"]))
    packed = np.load(dataset_path_string)
    dataset = tuple(packed[name] for name in ("states", "source_actions", "targets", "masks", "caps"))
    agent, _ = load_agent(resolve(contract["source_checkpoint"]))
    history, train_idx, validation_idx = train_branchwise(
        agent, dataset, candidate, contract["objective"]
    )
    validation = branch_metrics(
        agent, dataset, validation_idx, int(contract["budget"]),
        int(contract["objective"]["batch_size"]),
    )
    risk = validate_ch_risk_config(json.loads(resolve(contract["risk_config"]).read_text()))
    qos = Step3QoSConstraintConfig.from_payload(json.loads(resolve(contract["qos_config"]).read_text()))
    environments, _, _ = build_environments(
        None, risk, int(contract["horizon"]), seeds=contract["development_seeds"]
    )
    result = evaluate(agent, environments, qos, contract)
    gates = contract["gates"]
    reduction = 1.0 - result["intervention"]["changed_slots_l1"] / max(
        1, baseline["intervention"]["changed_slots_l1"]
    )
    checks = {
        "joint_qos": result["joint_qos_pass_count"] >= int(gates["minimum_joint_qos_pairs"]),
        "intervention_reduction": reduction >= float(gates["minimum_intervention_slot_reduction_fraction"]),
        "fnd": result["mean_fnd_free_steps"] >= baseline["mean_fnd_free_steps"] - float(gates["maximum_mean_fnd_degradation_rounds"]),
        "fairness": result["mean_episode_service_fairness"] >= baseline["mean_episode_service_fairness"] - float(gates["maximum_fairness_degradation"]),
        "packets_per_j": result["mean_global_packets_per_j"] >= baseline["mean_global_packets_per_j"] * (1.0 - float(gates["maximum_packets_per_j_degradation_fraction"])),
        "unchanged_agreement": validation["unchanged_agreement"] >= float(gates["minimum_unchanged_branch_agreement"]),
    }
    output_dir = ROOT / "outputs" / "phase2" / "step3_primary_branchwise_distillation_sweep_v1" / candidate["candidate_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "branching_c51_branchwise_distilled.pt"
    agent.save(checkpoint, {"method": "primary_branchwise_intervention_distillation", "candidate": candidate})
    return {
        "candidate": candidate, "history": history,
        "validation_branch_metrics": validation, "evaluation": result,
        "intervention_reduction_fraction": reduction, "checks": checks,
        "gate_pass": all(checks.values()),
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
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
    environments, _, _ = build_environments(None, risk, int(contract["horizon"]), seeds=contract["development_seeds"])
    states, source_actions, targets, masks, caps, collection = collect(source, environments, contract)
    output_root = ROOT / "outputs" / "phase2" / "step3_primary_branchwise_distillation_sweep_v1"
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_path = output_root / "demonstrations.npz"
    np.savez_compressed(dataset_path, states=states, source_actions=source_actions, targets=targets, masks=masks, caps=caps)
    baseline_envs, _, _ = build_environments(None, risk, int(contract["horizon"]), seeds=contract["development_seeds"])
    baseline = evaluate(source, baseline_envs, qos, contract)
    candidates = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(contract["parallel_candidates"])) as pool:
        futures = [pool.submit(candidate_worker, str(contract_path), str(dataset_path), candidate, baseline) for candidate in contract["candidates"]]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            candidates.append(row)
            print(f"CANDIDATE_COMPLETE={row['candidate']['candidate_id']} PASS={row['gate_pass']}", flush=True)
    order = {candidate["candidate_id"]: index for index, candidate in enumerate(contract["candidates"])}
    candidates.sort(key=lambda row: order[row["candidate"]["candidate_id"]])
    passing = [row for row in candidates if row["gate_pass"]]
    selected = min(passing, key=lambda row: (
        row["evaluation"]["intervention"]["changed_slots_l1"],
        -row["validation_branch_metrics"]["unchanged_agreement"],
        -row["evaluation"]["mean_episode_service_fairness"],
        -row["evaluation"]["mean_global_packets_per_j"],
        order[row["candidate"]["candidate_id"]],
    )) if passing else None
    payload = {
        "schema_version": 1,
        "status": "primary_branchwise_candidate_selected" if selected else "no_primary_branchwise_candidate_passed",
        "contract_sha256": sha256(contract_path), "evaluator_sha256": sha256(Path(__file__)),
        "collection": collection, "demonstrations": len(states), "baseline": baseline,
        "candidates": candidates,
        "selected_candidate_id": selected["candidate"]["candidate_id"] if selected else None,
        "selected_checkpoint": selected["checkpoint"] if selected else None,
        "selected_checkpoint_sha256": selected["checkpoint_sha256"] if selected else None,
        "longer_training_authorized": bool(selected), "confirmation_seeds_opened": False,
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
