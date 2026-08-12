"""No-learning diagnosis of the Step 3 target-delivery deficit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.train_phase2_dynamic_curriculum as trainer
from agents.branching_dqn import BranchingAgentConfig, BranchingDQNAgent
from agents.ch_depletion_risk import validate_ch_risk_config
from agents.qos_constraints_v3 import Step3QoSConstraintConfig
from envs.fixed_cluster_training_env import FixedClusterTrainingEnv
from envs.step3_lifetime_env import RoleSeparatedScheduledMACEnv, configure_step3_risk
from envs.step3_policy_observation import STEP3_CH_CONTEXT_SCHEMA
from envs.step3_v3_env import Step3V3DynamicClusterTrainingEnv


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fair_budget_fill(caps, mask, budget, cumulative_service):
    """Maximize immediate feasible delivery, breaking ties toward fairness."""
    caps = np.asarray(caps, dtype=np.int64)
    mask = np.asarray(mask, dtype=bool)
    service = np.asarray(cumulative_service, dtype=np.float64)
    allocation = np.zeros_like(caps)
    for _ in range(int(budget)):
        eligible = mask & (allocation < caps)
        if not np.any(eligible):
            break
        candidates = np.flatnonzero(eligible)
        remaining = caps[candidates] - allocation[candidates]
        order = np.lexsort((candidates, -remaining, service[candidates] + allocation[candidates]))
        allocation[int(candidates[order[0]])] += 1
    return allocation


def unconstrained_argmax(q_values, caps, mask):
    q = np.asarray(q_values, dtype=np.float64)
    caps = np.asarray(caps, dtype=np.int64)
    mask = np.asarray(mask, dtype=bool)
    action = np.zeros(len(mask), dtype=np.int64)
    for node in np.flatnonzero(mask):
        action[node] = int(np.argmax(q[node, : int(caps[node]) + 1]))
    return action


def load_agent(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = BranchingAgentConfig(**dict(checkpoint["config"]))
    agent = BranchingDQNAgent(config, device="cpu")
    agent.online.load_state_dict(checkpoint["online_state_dict"])
    agent.target.load_state_dict(checkpoint["target_state_dict"])
    agent.online.eval()
    return agent, checkpoint


def build_environments(
    profile_path: Path | None, risk_config: dict, horizon: int, seeds=(2400,),
):
    configure_step3_risk(risk_config)
    trainer.ScheduledIntraClusterMACEnv = RoleSeparatedScheduledMACEnv
    trainer.DynamicClusterTrainingEnv = Step3V3DynamicClusterTrainingEnv
    return trainer.build_curriculum(
        [int(seed) for seed in seeds], horizon,
        observation_schema=STEP3_CH_CONTEXT_SCHEMA,
        environment_profile=profile_path,
    )


def evaluate_policy(policy_name, environments, agent, qos, budget):
    rows = []
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
        steps = 0
        done = False
        previous_transition = False
        while not done:
            current_members = env.members.copy()
            current_ch = int(env.ch)
            alive_members = env.base.alive[current_members]
            ch_alive = bool(env.base.alive[current_ch])
            padded, padded_mask, caps = trainer.padded_state(
                env, observation, mask, env.base.n_nodes
            )
            feasible = int(caps.sum()) if ch_alive else 0
            offered = int(env.base.queue[current_members][alive_members].sum()) if ch_alive else 0
            oracle_delivery = min(feasible, int(budget))

            if policy_name == "trained_greedy":
                action, q_values = agent.act(
                    padded, padded_mask, epsilon=0.0, caps=caps,
                    budget=budget, tie_break_priorities=np.arange(env.base.n_nodes),
                )
                requested = unconstrained_argmax(q_values, caps, padded_mask)
            elif policy_name == "fair_budget_fill_oracle":
                action = fair_budget_fill(
                    caps, padded_mask, budget, env.cumulative_service
                )
                requested = action.copy()
            else:
                raise ValueError(policy_name)

            requested_service = int(np.minimum(caps, requested).sum())
            projected_slots = int(action.sum())
            next_observation, next_mask, done, info = env.step(action)
            executed = int(info["target_packets_delivered"])
            role = info["energy_trace"]["role_energy"]
            ch_forward = (
                float(role["ch_rx"][current_ch])
                + float(role["ch_aggregate"][current_ch])
                + float(role["ch_tx_bs"][current_ch])
            )
            next_members = env.members.copy()
            next_ch = int(env.ch)
            member_transition = not np.array_equal(current_members, next_members)
            ch_transition = current_ch != next_ch
            service_gap = max(0, oracle_delivery - executed)

            counters["offered_demand"] += offered
            counters["feasible_backlog"] += feasible
            counters["budget_feasible_delivery"] += oracle_delivery
            counters["requested_slots"] += int(requested.sum())
            counters["requested_service"] += requested_service
            counters["projected_slots"] += projected_slots
            counters["executed_delivery"] += executed
            counters["unused_budget"] += max(0, int(budget) - projected_slots)
            counters["service_gap_to_budget_oracle"] += service_gap
            counters["stale_drops"] += int(info["target_stale_drops"])
            counters["empty_target_steps"] += int(len(current_members) == 0)
            counters["dead_ch_before_steps"] += int(not ch_alive)
            counters["ch_death_terminations"] += int(ch_alive and not env.base.alive[current_ch])
            counters["membership_transitions"] += int(member_transition)
            counters["ch_transitions"] += int(ch_transition)
            counters["post_transition_service_gap"] += int(previous_transition) * service_gap
            counters["member_tx_energy_j"] += float(role["member_tx"][current_members].sum())
            counters["ch_forwarding_energy_j"] += ch_forward
            previous_transition = member_transition or ch_transition
            observation, mask = next_observation, next_mask
            steps += 1

        demand = max(1, counters["offered_demand"])
        delivery_ratio = counters["executed_delivery"] / demand
        stale_ratio = counters["stale_drops"] / demand
        fairness = float(env.step3_qos_counts["fairness"])
        episode_fairness = float(
            env.step3_qos_counts.get("episode_service_fairness", fairness)
        )
        joint = (
            delivery_ratio >= qos.minimum_delivery_ratio
            and stale_ratio <= qos.maximum_stale_drop_ratio
            and fairness >= qos.minimum_queue_fairness
        )
        rows.append({
            "seed": env.seed,
            "target_rank": env.target_rank,
            "steps": steps,
            **counters,
            "delivery_ratio": delivery_ratio,
            "stale_ratio": stale_ratio,
            "fairness": fairness,
            "episode_service_fairness": episode_fairness,
            "joint_qos_pass": bool(joint),
            "fnd_free_steps": int(env.base.t_fnd if env.base.t_fnd is not None else steps),
            "target_ch_alive_at_end": bool(env.base.alive[int(env.ch)]),
        })
    return rows


def aggregate(rows, qos):
    keys = [
        "offered_demand", "feasible_backlog", "budget_feasible_delivery",
        "requested_slots", "requested_service", "projected_slots",
        "executed_delivery", "unused_budget", "service_gap_to_budget_oracle",
        "stale_drops", "empty_target_steps", "dead_ch_before_steps",
        "ch_death_terminations", "membership_transitions", "ch_transitions",
        "post_transition_service_gap", "member_tx_energy_j", "ch_forwarding_energy_j",
    ]
    totals = {key: float(sum(row[key] for row in rows)) for key in keys}
    total_demand = max(1.0, totals["offered_demand"])
    return {
        "pairs": len(rows),
        "joint_qos_pass_count": int(sum(row["joint_qos_pass"] for row in rows)),
        "delivery_pass_count": int(sum(row["delivery_ratio"] >= qos.minimum_delivery_ratio for row in rows)),
        "stale_pass_count": int(sum(row["stale_ratio"] <= qos.maximum_stale_drop_ratio for row in rows)),
        "fairness_pass_count": int(sum(row["fairness"] >= qos.minimum_queue_fairness for row in rows)),
        "macro_mean_delivery_ratio": float(np.mean([row["delivery_ratio"] for row in rows])),
        "micro_delivery_ratio": totals["executed_delivery"] / total_demand,
        "mean_fnd_free_steps": float(np.mean([row["fnd_free_steps"] for row in rows])),
        "mean_fairness": float(np.mean([row["fairness"] for row in rows])),
        "mean_episode_service_fairness": float(
            np.mean([row["episode_service_fairness"] for row in rows])
        ),
        "totals": totals,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--environment-profile", type=Path, required=True)
    parser.add_argument("--ch-risk-config", type=Path, required=True)
    parser.add_argument("--qos-config", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=1200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.horizon != 1200:
        raise ValueError("this frozen diagnostic requires horizon 1200")
    resolve = lambda path: path if path.is_absolute() else ROOT / path
    checkpoint_path = resolve(args.checkpoint)
    profile_path = resolve(args.environment_profile)
    risk_path = resolve(args.ch_risk_config)
    qos_path = resolve(args.qos_config)
    output_path = resolve(args.output)
    risk = validate_ch_risk_config(json.loads(risk_path.read_text()))
    qos = Step3QoSConstraintConfig.from_payload(json.loads(qos_path.read_text()))
    agent, checkpoint = load_agent(checkpoint_path)
    if agent.cfg.state_schema != STEP3_CH_CONTEXT_SCHEMA or agent.cfg.budget != 16:
        raise RuntimeError("checkpoint does not obey the frozen Step 3 v3 B16 contract")

    results = {}
    manifest = None
    for name in ("trained_greedy", "fair_budget_fill_oracle"):
        environments, manifest, cfg = build_environments(profile_path, risk, args.horizon)
        rows = evaluate_policy(name, environments, agent, qos, cfg.frame_slot_budget)
        results[name] = {"aggregate": aggregate(rows, qos), "pairs": rows}
    oracle = results["fair_budget_fill_oracle"]["aggregate"]
    required_pairs = int(np.ceil(0.90 * oracle["pairs"]))
    reachable = oracle["joint_qos_pass_count"] >= required_pairs
    payload = {
        "schema_version": 1,
        "status": "delivery_floor_structurally_reachable" if reachable else "delivery_floor_not_reachable_stop_training",
        "development_only": True,
        "no_learning": True,
        "seed": 2400,
        "target_ranks": list(range(len(manifest))),
        "horizon": args.horizon,
        "budget": 16,
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_episode": checkpoint.get("metadata", {}).get("episode"),
        "environment_profile_sha256": sha256(profile_path),
        "risk_config_sha256": sha256(risk_path),
        "qos_config_sha256": sha256(qos_path),
        "thresholds": {
            "delivery_min": qos.minimum_delivery_ratio,
            "stale_max": qos.maximum_stale_drop_ratio,
            "fairness_min": qos.minimum_queue_fairness,
            "required_joint_fraction": 0.90,
            "required_joint_pairs": required_pairs,
        },
        "ch_forwarding_capacity_contract": "no_packet_capacity_cap_in_environment; forwarding cost is energy-only",
        "results": results,
        "oracle_reachability_pass": bool(reachable),
        "next_training_candidate_authorized": bool(reachable),
        "claim_boundary": "development_no_learning_reachability_diagnostic_not_model_performance",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "trained": results["trained_greedy"]["aggregate"],
        "oracle": oracle,
        "output": str(output_path),
    }, indent=2))
    raise SystemExit(0 if reachable else 3)


if __name__ == "__main__":
    main()
