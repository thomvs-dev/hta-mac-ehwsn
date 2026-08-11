"""Development-only DQfD-style distillation of the frozen Step 3 QoS shield."""

from __future__ import annotations

import argparse
import json
import os
import random
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
from envs.step3_policy_observation import STEP3_CH_CONTEXT_SCHEMA
from experiments.diagnose_step3_delivery_feasibility import (
    aggregate,
    build_environments,
    evaluate_policy,
    load_agent,
    sha256,
)
from experiments.sweep_step3_qos_deficit_override import qos_deficit_override
from experiments.sweep_step3_risk_gated_completion import forwarding_energy_j


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def set_cpu_contract(threads: int, seed: int) -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, min(4, threads // 4)))
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)


def collect_intervention_demonstrations(agent, environments, budget, controller):
    states, actions, masks, caps_rows = [], [], [], []
    totals = {"steps": 0, "intervention_steps": 0, "added_slots": 0, "risk_blocked_slots": 0}
    agent.online.eval()
    for env in environments:
        observation, mask, _ = env.reset()
        done = False
        while not done:
            padded, padded_mask, caps = trainer.padded_state(
                env, observation, mask, env.base.n_nodes
            )
            base_action, q_values = agent.act(
                padded, padded_mask, epsilon=0.0, caps=caps, budget=budget,
                tie_break_priorities=np.arange(env.base.n_nodes),
            )
            demonstrated, audit = qos_deficit_override(
                base_action, q_values, caps, padded_mask, env,
                trajectory_target=controller["delivery_trajectory_target"],
                reserve_floor=controller["ch_post_forwarding_reserve_floor"],
                completion_fraction=controller["deficit_completion_fraction"],
            )
            if not np.array_equal(base_action, demonstrated):
                states.append(np.asarray(padded, dtype=np.float16))
                actions.append(np.asarray(demonstrated, dtype=np.uint8))
                masks.append(np.asarray(padded_mask, dtype=bool))
                caps_rows.append(np.asarray(caps, dtype=np.uint8))
                totals["intervention_steps"] += 1
            totals["steps"] += 1
            totals["added_slots"] += int(audit["added"])
            totals["risk_blocked_slots"] += int(audit["risk_blocked"])
            observation, mask, done, _ = env.step(demonstrated)
    if not states:
        raise RuntimeError("shield produced no intervention demonstrations")
    return (
        np.stack(states), np.stack(actions), np.stack(masks), np.stack(caps_rows), totals
    )


def margin_loss(agent, states, actions, masks, caps, margin):
    model_states = agent._transform_state_tensor(states)
    log_probabilities = agent.online(model_states, masks)
    q_values = (log_probabilities.exp() * agent.online.support).sum(dim=-1)
    chosen_q = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)
    levels = torch.arange(agent.cfg.actions, device=agent.device).view(1, 1, -1)
    feasible = levels <= caps.unsqueeze(-1)
    additive = (levels != actions.unsqueeze(-1)).to(q_values.dtype) * float(margin)
    competitors = (q_values + additive).masked_fill(~feasible, -torch.inf)
    per_branch = torch.relu(competitors.max(dim=-1).values - chosen_q)
    valid = masks.float()
    return ((per_branch * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)).mean()


@torch.no_grad()
def agreement(agent, states_np, actions_np, masks_np, caps_np, budget, batch_size):
    exact = 0
    absolute_error = 0
    count = len(states_np)
    agent.online.eval()
    for start in range(0, count, batch_size):
        stop = min(count, start + batch_size)
        states = torch.as_tensor(states_np[start:stop], dtype=torch.float32, device=agent.device)
        masks = torch.as_tensor(masks_np[start:stop], dtype=torch.bool, device=agent.device)
        q = agent.online.q_values(agent._transform_state_tensor(states), masks).cpu().numpy()
        for offset, values in enumerate(q):
            index = start + offset
            predicted = agent._project(values, masks_np[index], caps=caps_np[index])
            target = actions_np[index].astype(np.int64)
            exact += int(np.array_equal(predicted, target))
            absolute_error += int(np.abs(predicted - target).sum())
    return {
        "exact_action_fraction": exact / count,
        "mean_branch_absolute_error": absolute_error / (count * actions_np.shape[1]),
        "samples": count,
    }


def train_distillation(agent, dataset, *, epochs, batch_size, learning_rate, margin, seed):
    states_np, actions_np, masks_np, caps_np = dataset
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(states_np))
    split = max(1, int(0.8 * len(order)))
    train_idx, validation_idx = order[:split], order[split:]
    if not len(validation_idx):
        validation_idx = train_idx[-1:]
    optimizer = torch.optim.Adam(agent.online.parameters(), lr=learning_rate, weight_decay=1e-5)
    history = []
    for epoch in range(1, epochs + 1):
        agent.online.train()
        epoch_losses = []
        for start in range(0, len(train_idx), batch_size):
            batch_idx = train_idx[start:start + batch_size]
            states = torch.as_tensor(states_np[batch_idx], dtype=torch.float32, device=agent.device)
            actions = torch.as_tensor(actions_np[batch_idx], dtype=torch.long, device=agent.device)
            masks = torch.as_tensor(masks_np[batch_idx], dtype=torch.bool, device=agent.device)
            caps = torch.as_tensor(caps_np[batch_idx], dtype=torch.long, device=agent.device)
            loss = margin_loss(agent, states, actions, masks, caps, margin)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.online.parameters(), 10.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        train_metrics = agreement(
            agent, states_np[train_idx], actions_np[train_idx], masks_np[train_idx],
            caps_np[train_idx], agent.cfg.budget, batch_size,
        )
        validation_metrics = agreement(
            agent, states_np[validation_idx], actions_np[validation_idx],
            masks_np[validation_idx], caps_np[validation_idx], agent.cfg.budget, batch_size,
        )
        row = {
            "epoch": epoch,
            "mean_margin_loss": float(np.mean(epoch_losses)),
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(row)
        print(
            f"DISTILL_EPOCH={epoch}/{epochs} LOSS={row['mean_margin_loss']:.6f} "
            f"TRAIN_EXACT={train_metrics['exact_action_fraction']:.4f} "
            f"VALID_EXACT={validation_metrics['exact_action_fraction']:.4f}",
            flush=True,
        )
    agent.target.load_state_dict(agent.online.state_dict())
    agent.optimizer = optimizer
    return history, train_idx, validation_idx


def controller_on_evaluation(agent, environments, qos, budget, controller):
    rows = []
    for env in environments:
        observation, mask, _ = env.reset()
        done = False
        while not done:
            padded, padded_mask, caps = trainer.padded_state(env, observation, mask, env.base.n_nodes)
            base_action, q_values = agent.act(
                padded, padded_mask, epsilon=0.0, caps=caps, budget=budget,
                tie_break_priorities=np.arange(env.base.n_nodes),
            )
            action, _ = qos_deficit_override(
                base_action, q_values, caps, padded_mask, env,
                trajectory_target=controller["delivery_trajectory_target"],
                reserve_floor=controller["ch_post_forwarding_reserve_floor"],
                completion_fraction=controller["deficit_completion_fraction"],
            )
            observation, mask, done, _ = env.step(action)
        counts = env.step3_qos_counts
        demand = max(1, int(counts["demand"]))
        delivery = int(counts["delivered"]) / demand
        stale = int(counts["stale"]) / demand
        fairness = float(counts["fairness"])
        rows.append({
            "delivery_ratio": delivery,
            "stale_ratio": stale,
            "fairness": fairness,
            "joint_qos_pass": bool(
                delivery >= qos.minimum_delivery_ratio
                and stale <= qos.maximum_stale_drop_ratio
                and fairness >= qos.minimum_queue_fairness
            ),
            "fnd_free_steps": int(env.base.t_fnd if env.base.t_fnd is not None else 1200),
            **{key: 0 for key in (
                "offered_demand", "feasible_backlog", "budget_feasible_delivery",
                "requested_slots", "requested_service", "projected_slots",
                "executed_delivery", "unused_budget", "service_gap_to_budget_oracle",
                "stale_drops", "empty_target_steps", "dead_ch_before_steps",
                "ch_death_terminations", "membership_transitions", "ch_transitions",
                "post_transition_service_gap", "member_tx_energy_j", "ch_forwarding_energy_j",
            )},
        })
    return {
        "pairs": len(rows),
        "joint_qos_pass_count": int(sum(row["joint_qos_pass"] for row in rows)),
        "delivery_pass_count": int(sum(row["delivery_ratio"] >= qos.minimum_delivery_ratio for row in rows)),
        "stale_pass_count": int(sum(row["stale_ratio"] <= qos.maximum_stale_drop_ratio for row in rows)),
        "fairness_pass_count": int(sum(row["fairness"] >= qos.minimum_queue_fairness for row in rows)),
        "macro_mean_delivery_ratio": float(np.mean([row["delivery_ratio"] for row in rows])),
        "mean_fnd_free_steps": float(np.mean([row["fnd_free_steps"] for row in rows])),
        "mean_fairness": float(np.mean([row["fairness"] for row in rows])),
        "micro_delivery_ratio": None,
        "micro_delivery_ratio_note": "not collected by the compact controller-on evaluator",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--environment-profile", type=Path, required=True)
    parser.add_argument("--ch-risk-config", type=Path, required=True)
    parser.add_argument("--qos-config", type=Path, required=True)
    parser.add_argument("--controller-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--margin", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=5701)
    parser.add_argument("--cpu-threads", type=int, default=18)
    parser.add_argument("--horizon", type=int, default=1200)
    args = parser.parse_args()
    if args.epochs != 5 or args.batch_size != 64 or args.learning_rate != 3e-5 or args.margin != 0.8:
        raise RuntimeError("frozen shield-distillation parameter contract changed")
    if args.seed != 5701 or args.horizon != 1200:
        raise RuntimeError("frozen development seed/horizon contract changed")
    set_cpu_contract(args.cpu_threads, args.seed)
    paths = {name: resolve(getattr(args, name)) for name in (
        "checkpoint", "environment_profile", "ch_risk_config", "qos_config", "controller_config", "output_dir"
    )}
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    controller = json.loads(paths["controller_config"].read_text())
    risk = validate_ch_risk_config(json.loads(paths["ch_risk_config"].read_text()))
    qos = Step3QoSConstraintConfig.from_payload(json.loads(paths["qos_config"].read_text()))
    agent, source_checkpoint = load_agent(paths["checkpoint"])
    if agent.cfg.state_schema != STEP3_CH_CONTEXT_SCHEMA or agent.cfg.budget != 16:
        raise RuntimeError("source checkpoint is not the frozen Step 3 v3 B16 model")
    started = time.perf_counter()
    environments, manifest, cfg = build_environments(paths["environment_profile"], risk, args.horizon)
    states, actions, masks, caps, collection = collect_intervention_demonstrations(
        agent, environments, cfg.frame_slot_budget, controller
    )
    baseline_agreement = agreement(agent, states, actions, masks, caps, cfg.frame_slot_budget, args.batch_size)
    history, train_idx, validation_idx = train_distillation(
        agent, (states, actions, masks, caps), epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        margin=args.margin, seed=args.seed,
    )
    checkpoint_path = paths["output_dir"] / "branching_c51_shield_distilled.pt"
    agent.save(checkpoint_path, {
        "method": "dqfd_style_large_margin_shield_distillation",
        "source_checkpoint_sha256": sha256(paths["checkpoint"]),
        "development_seed": 2400,
        "optimizer_seed": args.seed,
        "epochs": args.epochs,
    })
    off_envs, _, _ = build_environments(paths["environment_profile"], risk, args.horizon)
    controller_off_rows = evaluate_policy(
        "trained_greedy", off_envs, agent, qos, cfg.frame_slot_budget
    )
    on_envs, _, _ = build_environments(paths["environment_profile"], risk, args.horizon)
    controller_on = controller_on_evaluation(agent, on_envs, qos, cfg.frame_slot_budget, controller)
    payload = {
        "schema_version": 1,
        "status": "adaptive_development_distillation_complete",
        "method": "dqfd_style_large_margin_distillation_from_shield_interventions",
        "publication_evidence": False,
        "confirmation_evidence": False,
        "development_seed": 2400,
        "optimizer_seed": args.seed,
        "source_checkpoint_sha256": sha256(paths["checkpoint"]),
        "output_checkpoint_sha256": sha256(checkpoint_path),
        "parameters": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "margin": args.margin,
            "cpu_threads": args.cpu_threads,
        },
        "collection": collection,
        "demonstrations": int(len(states)),
        "baseline_demonstration_agreement": baseline_agreement,
        "history": history,
        "controller_off": aggregate(controller_off_rows, qos),
        "controller_on": controller_on,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": "single adaptive development seed; requires independent frozen confirmation",
    }
    report_path = paths["output_dir"] / "shield_distillation_report.json"
    report_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"REPORT={report_path}", flush=True)
    print(f"CHECKPOINT={checkpoint_path}", flush=True)
    print(f"CONTROLLER_OFF_QOS={payload['controller_off']['joint_qos_pass_count']}/20", flush=True)
    print(f"CONTROLLER_ON_QOS={payload['controller_on']['joint_qos_pass_count']}/20", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
