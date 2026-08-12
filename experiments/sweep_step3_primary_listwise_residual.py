"""Permutation-equivariant listwise residual ranking with on-policy aggregation."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.train_phase2_dynamic_curriculum as trainer
from agents.ch_depletion_risk import validate_ch_risk_config
from agents.qos_constraints_v3 import Step3QoSConstraintConfig
from experiments.diagnose_step3_delivery_feasibility import build_environments, load_agent
from experiments.distill_step3_qos_shield import set_cpu_contract
from experiments.sweep_step3_qos_band_projection import qos_band_projection
from experiments.sweep_step3_qos_deficit_override import qos_deficit_override


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha256(path: str | Path) -> str:
    return hashlib.sha256(resolve(path).read_bytes()).hexdigest()


def load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text())
    if contract.get("status") not in {
        "frozen_before_primary_listwise_residual_sweep",
        "frozen_before_primary_listwise_residual_continuation",
    }:
        raise RuntimeError("listwise residual contract is not frozen")
    for field in ("source_checkpoint", "qos_config", "risk_config"):
        if sha256(contract[field]) != contract[f"{field}_sha256"]:
            raise RuntimeError(f"artifact hash mismatch: {field}")
    if contract.get("source_ranker_checkpoint") and (
        sha256(contract["source_ranker_checkpoint"])
        != contract["source_ranker_checkpoint_sha256"]
    ):
        raise RuntimeError("artifact hash mismatch: source_ranker_checkpoint")
    if set(contract["development_seeds"]).intersection(contract["prohibited_seeds"]):
        raise RuntimeError("prohibited seed requested")
    return contract


class SetRemovalRanker(nn.Module):
    """Shared node scorer; score permutation follows the input permutation."""

    def __init__(self, features: int, hidden: int):
        super().__init__()
        self.features = int(features)
        self.hidden = int(hidden)
        self.scorer = (
            nn.Linear(features, 1)
            if hidden <= 0 else
            nn.Sequential(nn.Linear(features, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        )

    def forward(self, features):
        return self.scorer(features).squeeze(-1)


def removal_count(action, env, band):
    counts = env.step3_qos_counts
    offered = int(env.base.queue[env.members][env.base.alive[env.members]].sum())
    predicted_demand = int(counts["demand"]) + offered
    lower_required = int(math.ceil(float(band["lower_delivery_target"]) * predicted_demand))
    upper_allowed = int(math.floor(float(band["upper_delivery_target"]) * predicted_demand))
    predicted_delivery = int(counts["delivered"]) + int(action.sum())
    return min(max(0, predicted_delivery - upper_allowed), max(0, predicted_delivery - lower_required))


def ranking_features(env, q_values, action, caps, mask):
    active = np.asarray(mask, dtype=bool) & (np.asarray(action) > 0)
    service = np.asarray(env.cumulative_service, dtype=np.float64)
    active_service = service[active]
    percentile = np.zeros_like(service)
    fraction = np.zeros_like(service)
    if active_service.size:
        order = np.argsort(np.argsort(active_service, kind="stable"), kind="stable")
        percentile[active] = order / max(1, active_service.size - 1)
        fraction[active] = active_service / max(1.0, float(active_service.max()))
    marginal = np.zeros_like(service)
    for node in np.flatnonzero(active):
        level = int(action[node])
        marginal[node] = float(q_values[node, level] - q_values[node, level - 1])
    standardized = np.zeros_like(service)
    if active_service.size:
        values = -marginal[active]
        standardized[active] = (values - values.mean()) / max(1e-6, values.std())
    n_max = max(1, q_values.shape[1] - 1)
    features = np.stack((
        percentile, fraction, standardized,
        np.asarray(action, dtype=np.float64) / n_max,
        np.asarray(caps, dtype=np.float64) / n_max,
    ), axis=-1).astype(np.float32)
    return features, active


def teacher_winner(env, q_values, action, mask):
    candidates = []
    service = np.asarray(env.cumulative_service, dtype=np.float64)
    for node in np.flatnonzero(np.asarray(mask, dtype=bool) & (action > 0)):
        level = int(action[node])
        q_loss = float(q_values[node, level] - q_values[node, level - 1])
        candidates.append((float(service[node]), -q_loss, -int(node), int(node)))
    if not candidates:
        raise RuntimeError("removal requested without an eligible branch")
    return max(candidates)[-1]


def base_and_added(agent, env, observation, mask, contract):
    padded, padded_mask, caps = trainer.padded_state(env, observation, mask, env.base.n_nodes)
    base, q_values = agent.act(
        padded, padded_mask, epsilon=0.0, caps=caps, budget=int(contract["budget"]),
        tie_break_priorities=np.arange(env.base.n_nodes),
    )
    band = contract["band"]
    added, _ = qos_deficit_override(
        base, q_values, caps, padded_mask, env,
        trajectory_target=float(band["lower_delivery_target"]),
        reserve_floor=float(band["reserve_floor"]),
        completion_fraction=float(band["completion_fraction"]),
    )
    return padded, padded_mask, caps, base, q_values, added


def collect_samples(agent, environments, contract, ranker=None):
    feature_rows, eligible_rows, winner_rows = [], [], []
    trajectory = {"steps": 0, "teacher_removals": 0, "ranker_actions": 0}
    for env in environments:
        observation, mask, _ = env.reset()
        done = False
        while not done:
            _, padded_mask, caps, _, q_values, added = base_and_added(agent, env, observation, mask, contract)
            teacher_action = added.copy()
            count = removal_count(teacher_action, env, contract["band"])
            for _ in range(count):
                features, eligible = ranking_features(env, q_values, teacher_action, caps, padded_mask)
                winner = teacher_winner(env, q_values, teacher_action, padded_mask)
                feature_rows.append(features)
                eligible_rows.append(eligible)
                winner_rows.append(winner)
                teacher_action[winner] -= 1
                trajectory["teacher_removals"] += 1
            if ranker is None:
                action = teacher_action
            else:
                action = apply_ranker(ranker, env, q_values, added, caps, padded_mask, count)
                trajectory["ranker_actions"] += 1
            trajectory["steps"] += 1
            observation, mask, done, _ = env.step(action)
    return (
        np.stack(feature_rows), np.stack(eligible_rows), np.asarray(winner_rows, dtype=np.int64), trajectory
    )


@torch.no_grad()
def apply_ranker(ranker, env, q_values, action, caps, mask, count):
    result = np.asarray(action, dtype=np.int64).copy()
    ranker.eval()
    for _ in range(int(count)):
        features, eligible = ranking_features(env, q_values, result, caps, mask)
        if not eligible.any():
            break
        tensor = torch.as_tensor(features, dtype=torch.float32)
        scores = ranker(tensor).masked_fill(~torch.as_tensor(eligible), -torch.inf)
        result[int(scores.argmax().item())] -= 1
    return result


@torch.no_grad()
def top1_accuracy(ranker, dataset, indices, batch_size):
    features, eligible, winners = dataset
    correct = 0
    ranker.eval()
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        x = torch.as_tensor(features[selected], dtype=torch.float32)
        mask = torch.as_tensor(eligible[selected], dtype=torch.bool)
        scores = ranker(x).masked_fill(~mask, -torch.inf)
        correct += int((scores.argmax(dim=1) == torch.as_tensor(winners[selected])).sum())
    return correct / max(1, len(indices))


def fit_ranker(ranker, dataset, candidate, batch_size, seed_offset=0):
    features, eligible, winners = dataset
    rng = np.random.default_rng(int(candidate["seed"]) + seed_offset)
    order = rng.permutation(len(features))
    split = max(1, int(0.8 * len(order)))
    train_idx, validation_idx = order[:split], order[split:]
    if not len(validation_idx):
        validation_idx = train_idx[-1:]
    optimizer = torch.optim.Adam(ranker.parameters(), lr=float(candidate["learning_rate"]), weight_decay=1e-5)
    history = []
    for epoch in range(1, int(candidate["epochs"]) + 1):
        shuffled = rng.permutation(train_idx)
        losses = []
        ranker.train()
        for start in range(0, len(shuffled), batch_size):
            selected = shuffled[start:start + batch_size]
            x = torch.as_tensor(features[selected], dtype=torch.float32)
            mask = torch.as_tensor(eligible[selected], dtype=torch.bool)
            target = torch.as_tensor(winners[selected], dtype=torch.long)
            scores = ranker(x).masked_fill(~mask, -1e9)
            loss = F.cross_entropy(scores, target)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ranker.parameters(), 5.0); optimizer.step()
            losses.append(float(loss.item()))
        history.append({"epoch": epoch, "mean_listnet_loss": float(np.mean(losses)),
                        "validation_top1_accuracy": top1_accuracy(ranker, dataset, validation_idx, batch_size)})
    return history, validation_idx


def evaluate_ranker(agent, ranker, environments, qos, contract):
    rows = []
    disagreement = 0
    teacher_changed = 0
    for env in environments:
        observation, mask, _ = env.reset(); done = False; energy = 0.0
        while not done:
            _, padded_mask, caps, base, q_values, added = base_and_added(agent, env, observation, mask, contract)
            teacher, _ = qos_band_projection(
                base, q_values, caps, padded_mask, env,
                lower_target=float(contract["band"]["lower_delivery_target"]),
                upper_target=float(contract["band"]["upper_delivery_target"]),
                reserve_floor=float(contract["band"]["reserve_floor"]),
                completion_fraction=float(contract["band"]["completion_fraction"]),
            )
            count = removal_count(added, env, contract["band"])
            action = apply_ranker(ranker, env, q_values, added, caps, padded_mask, count)
            disagreement += int(np.abs(action - teacher).sum())
            teacher_changed += int(np.abs(base - teacher).sum())
            observation, mask, done, info = env.step(action)
            energy += float(np.asarray(info["energy_trace"]["consumed"]).sum())
        counts = env.step3_qos_counts; demand = max(1, int(counts["demand"]))
        delivery = int(counts["delivered"]) / demand; stale = int(counts["stale"]) / demand
        fairness = float(counts["episode_service_fairness"])
        rows.append({"seed": int(env.seed), "target_rank": int(env.target_rank),
                     "delivery_ratio": delivery, "stale_ratio": stale,
                     "episode_service_fairness": fairness,
                     "joint_qos_pass": bool(delivery >= qos.minimum_delivery_ratio and stale <= qos.maximum_stale_drop_ratio and fairness >= qos.minimum_queue_fairness),
                     "fnd_free_steps": int(env.base.t_fnd if env.base.t_fnd is not None else contract["horizon"]),
                     "global_packets_per_j": int(env.base.total_packets) / max(energy, 1e-12)})
    return {"pairs": len(rows), "joint_qos_pass_count": int(sum(r["joint_qos_pass"] for r in rows)),
            "mean_delivery_ratio": float(np.mean([r["delivery_ratio"] for r in rows])),
            "mean_stale_ratio": float(np.mean([r["stale_ratio"] for r in rows])),
            "mean_fnd_free_steps": float(np.mean([r["fnd_free_steps"] for r in rows])),
            "mean_episode_service_fairness": float(np.mean([r["episode_service_fairness"] for r in rows])),
            "mean_global_packets_per_j": float(np.mean([r["global_packets_per_j"] for r in rows])),
            "teacher_action_disagreement_l1": disagreement, "teacher_changed_slots_l1": teacher_changed,
            "rows": rows}


def evaluate_teacher(agent, environments, qos, contract):
    """Evaluate the frozen analytic QoS-band teacher on a paired cohort."""
    rows = []
    intervention = {"steps": 0, "intervention_steps": 0, "added_slots": 0,
                    "removed_slots": 0, "risk_blocked_slots": 0,
                    "changed_slots_l1": 0}
    for env in environments:
        observation, mask, _ = env.reset(); done = False; energy = 0.0
        while not done:
            _, padded_mask, caps, base, q_values, _ = base_and_added(
                agent, env, observation, mask, contract
            )
            band = contract["band"]
            action, audit = qos_band_projection(
                base, q_values, caps, padded_mask, env,
                lower_target=float(band["lower_delivery_target"]),
                upper_target=float(band["upper_delivery_target"]),
                reserve_floor=float(band["reserve_floor"]),
                completion_fraction=float(band["completion_fraction"]),
            )
            difference = int(np.abs(base - action).sum())
            intervention["steps"] += 1
            intervention["intervention_steps"] += int(difference > 0)
            intervention["added_slots"] += int(audit["added"])
            intervention["removed_slots"] += int(audit["removed"])
            intervention["risk_blocked_slots"] += int(audit["risk_blocked"])
            intervention["changed_slots_l1"] += difference
            observation, mask, done, info = env.step(action)
            energy += float(np.asarray(info["energy_trace"]["consumed"]).sum())
        counts = env.step3_qos_counts; demand = max(1, int(counts["demand"]))
        delivery = int(counts["delivered"]) / demand
        stale = int(counts["stale"]) / demand
        fairness = float(counts["episode_service_fairness"])
        rows.append({"seed": int(env.seed), "target_rank": int(env.target_rank),
                     "delivery_ratio": delivery, "stale_ratio": stale,
                     "episode_service_fairness": fairness,
                     "joint_qos_pass": bool(delivery >= qos.minimum_delivery_ratio and stale <= qos.maximum_stale_drop_ratio and fairness >= qos.minimum_queue_fairness),
                     "fnd_free_steps": int(env.base.t_fnd if env.base.t_fnd is not None else contract["horizon"]),
                     "global_packets": int(env.base.total_packets),
                     "network_energy_j": energy,
                     "global_packets_per_j": int(env.base.total_packets) / max(energy, 1e-12)})
    return {"pairs": len(rows), "joint_qos_pass_count": int(sum(r["joint_qos_pass"] for r in rows)),
            "mean_delivery_ratio": float(np.mean([r["delivery_ratio"] for r in rows])),
            "mean_stale_ratio": float(np.mean([r["stale_ratio"] for r in rows])),
            "mean_episode_service_fairness": float(np.mean([r["episode_service_fairness"] for r in rows])),
            "mean_fnd_free_steps": float(np.mean([r["fnd_free_steps"] for r in rows])),
            "mean_global_packets_per_j": float(np.mean([r["global_packets_per_j"] for r in rows])),
            "intervention": intervention, "rows": rows}


def candidate_worker(contract_path_string, dataset_path_string, output_root_string, candidate, baseline):
    contract = load_contract(resolve(contract_path_string)); set_cpu_contract(int(contract["threads_per_candidate"]), int(candidate["seed"]))
    packed = np.load(dataset_path_string); dataset = tuple(packed[name] for name in ("features", "eligible", "winners"))
    torch.manual_seed(int(candidate["seed"])); ranker = SetRemovalRanker(dataset[0].shape[-1], int(candidate["hidden"]))
    history = []
    validation_idx = np.arange(max(1, int(0.8 * len(dataset[0]))), len(dataset[0]))
    if contract.get("source_ranker_checkpoint"):
        source_ranker = torch.load(resolve(contract["source_ranker_checkpoint"]), map_location="cpu", weights_only=False)
        if int(source_ranker["hidden"]) != int(candidate["hidden"]):
            raise RuntimeError("continuation ranker architecture mismatch")
        ranker.load_state_dict(source_ranker["state_dict"])
    else:
        history, validation_idx = fit_ranker(ranker, dataset, candidate, int(contract["batch_size"]))
    agent, _ = load_agent(resolve(contract["source_checkpoint"]))
    if int(candidate["dagger_iterations"]):
        risk = validate_ch_risk_config(json.loads(resolve(contract["risk_config"]).read_text()))
        dagger_envs, _, _ = build_environments(None, risk, int(contract["horizon"]), seeds=contract["development_seeds"])
        extra = collect_samples(agent, dagger_envs, contract, ranker=ranker)
        dataset = tuple(np.concatenate((old, new), axis=0) for old, new in zip(dataset, extra[:3]))
        dagger_history, validation_idx = fit_ranker(ranker, dataset, candidate, int(contract["batch_size"]), seed_offset=10000)
        history.extend({"dagger": True, **row} for row in dagger_history)
    validation_accuracy = top1_accuracy(ranker, dataset, validation_idx, int(contract["batch_size"]))
    risk = validate_ch_risk_config(json.loads(resolve(contract["risk_config"]).read_text()))
    qos = Step3QoSConstraintConfig.from_payload(json.loads(resolve(contract["qos_config"]).read_text()))
    envs, _, _ = build_environments(None, risk, int(contract["horizon"]), seeds=contract["development_seeds"])
    result = evaluate_ranker(agent, ranker, envs, qos, contract)
    reduction = 1.0 - result["teacher_action_disagreement_l1"] / max(1, baseline["intervention"]["changed_slots_l1"])
    gates = contract["gates"]
    checks = {"joint_qos": result["joint_qos_pass_count"] >= int(gates["minimum_joint_qos_pairs"]),
              "teacher_correction_reduction": reduction >= float(gates["minimum_teacher_correction_reduction_fraction"]),
              "validation_top1": validation_accuracy >= float(gates["minimum_validation_top1_accuracy"]),
              "fnd": result["mean_fnd_free_steps"] >= baseline["mean_fnd_free_steps"] - float(gates["maximum_mean_fnd_degradation_rounds"]),
              "fairness": result["mean_episode_service_fairness"] >= baseline["mean_episode_service_fairness"] - float(gates["maximum_fairness_degradation"]),
              "packets_per_j": result["mean_global_packets_per_j"] >= baseline["mean_global_packets_per_j"] * (1.0 - float(gates["maximum_packets_per_j_degradation_fraction"]))}
    out = Path(output_root_string) / candidate["candidate_id"]
    out.mkdir(parents=True, exist_ok=True); checkpoint = out / "removal_ranker.pt"
    torch.save({"state_dict": ranker.state_dict(), "features": ranker.features, "hidden": ranker.hidden, "candidate": candidate}, checkpoint)
    return {"candidate": candidate, "history": history, "validation_top1_accuracy": validation_accuracy,
            "aggregated_samples": len(dataset[0]), "evaluation": result,
            "teacher_correction_reduction_fraction": reduction, "checks": checks,
            "gate_pass": all(checks.values()), "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256(checkpoint)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--contract", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    contract_path, output_path = resolve(args.contract), resolve(args.output); contract = load_contract(contract_path); started = time.perf_counter()
    risk = validate_ch_risk_config(json.loads(resolve(contract["risk_config"]).read_text())); qos = Step3QoSConstraintConfig.from_payload(json.loads(resolve(contract["qos_config"]).read_text()))
    agent, _ = load_agent(resolve(contract["source_checkpoint"])); envs, _, _ = build_environments(None, risk, int(contract["horizon"]), seeds=contract["development_seeds"])
    features, eligible, winners, collection = collect_samples(agent, envs, contract)
    out_root = output_path.parent; out_root.mkdir(parents=True, exist_ok=True); dataset_path = out_root / "initial_listwise_demonstrations.npz"
    np.savez_compressed(dataset_path, features=features, eligible=eligible, winners=winners)
    baseline_envs, _, _ = build_environments(None, risk, int(contract["horizon"]), seeds=contract["development_seeds"]); baseline = evaluate_teacher(agent, baseline_envs, qos, contract)
    candidates = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(contract["parallel_candidates"])) as pool:
        futures = [pool.submit(candidate_worker, str(contract_path), str(dataset_path), str(out_root), candidate, baseline) for candidate in contract["candidates"]]
        for future in concurrent.futures.as_completed(futures):
            row = future.result(); candidates.append(row); print(f"CANDIDATE_COMPLETE={row['candidate']['candidate_id']} PASS={row['gate_pass']}", flush=True)
    order = {c["candidate_id"]: i for i, c in enumerate(contract["candidates"])}; candidates.sort(key=lambda row: order[row["candidate"]["candidate_id"]]); passing = [row for row in candidates if row["gate_pass"]]
    selected = min(passing, key=lambda row: (row["evaluation"]["teacher_action_disagreement_l1"], -row["validation_top1_accuracy"], -row["evaluation"]["mean_episode_service_fairness"], -row["evaluation"]["mean_global_packets_per_j"], order[row["candidate"]["candidate_id"]])) if passing else None
    payload = {"schema_version": 1, "status": "primary_listwise_residual_selected" if selected else "no_primary_listwise_residual_passed", "contract_sha256": sha256(contract_path), "evaluator_sha256": sha256(Path(__file__)), "initial_samples": len(features), "collection": collection, "baseline": baseline, "candidates": candidates, "selected_candidate_id": selected["candidate"]["candidate_id"] if selected else None, "selected_checkpoint": selected["checkpoint"] if selected else None, "selected_checkpoint_sha256": selected["checkpoint_sha256"] if selected else None, "longer_training_authorized": bool(selected), "confirmation_seeds_opened": False, "elapsed_seconds": time.perf_counter() - started, "claim_boundary": contract["claim_boundary"]}
    output_path.parent.mkdir(parents=True, exist_ok=True); output_path.write_text(json.dumps(payload, indent=2) + "\n"); print(json.dumps({"status": payload["status"], "selected": payload["selected_candidate_id"]}, indent=2)); print(f"OUTPUT={output_path}"); return 0 if selected else 3


if __name__ == "__main__":
    raise SystemExit(main())
