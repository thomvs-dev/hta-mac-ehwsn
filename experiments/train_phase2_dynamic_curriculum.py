"""Train HTA-MAC on frozen per-round CH schedules with padded branches."""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.branching_dqn import BranchingAgentConfig, BranchingDQNAgent
from agents.reward_model import RewardModel, TERM_ORDER
from envs.dynamic_cluster_training_env import DynamicClusterTrainingEnv
from envs.scheduled_mac_env import ScheduledIntraClusterMACEnv
from experiments.run_phase3_pilot import build_assets, schedule_bundle
from experiments.train_phase2_fixed_cluster import (
    beta_for,
    contribution_balance,
    git_hash,
    load_reward_model,
    set_seeds,
    trajectory_q_check,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument(
        "--development-seeds",
        default="2300,2301,2302,2303,2304",
    )
    parser.add_argument("--optimizer-seed", type=int, default=2299)
    parser.add_argument("--run-name", default="dynamic_curriculum_600ep_dev5")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--collapse-window", type=int, default=50)
    parser.add_argument("--learn-every", type=int, default=4)
    parser.add_argument("--idle-weight", type=float)
    parser.add_argument("--death-weight", type=float)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--initial-checkpoint" )
    parser.add_argument("--projection-budget", type=int)
    return parser.parse_args()


def epsilon_for(
    episode: int, total_episodes: int, start: float, end: float
) -> float:
    anneal_episodes = max(1, int(total_episodes * 0.8))
    fraction = min(1.0, episode / anneal_episodes)
    return start + fraction * (end - start)


def build_curriculum(seeds, max_steps):
    frozen_policy, solar, thermal, radio, cfg, manifest = build_assets(max_steps)
    environments = []
    curriculum_manifest = []
    checkpoint_sha = manifest["checkpoint"]["sha256"]
    for seed in seeds:
        bundle, schedule_metadata = schedule_bundle(
            frozen_policy, seed, max_steps, checkpoint_sha
        )
        initial_ranks = len(bundle["schedule"][0]["cluster_heads"])
        for target_rank in range(initial_ranks):
            base = ScheduledIntraClusterMACEnv(
                cfg,
                radio,
                solar,
                thermal,
                idle_energy_enabled=True,
            )
            env = DynamicClusterTrainingEnv(
                base,
                bundle,
                seed=seed,
                target_rank=target_rank,
            )
            observation, _, _ = env.reset()
            environments.append(env)
            curriculum_manifest.append(
                {
                    "seed": seed,
                    "target_rank": target_rank,
                    "initial_cluster": env.target_cluster,
                    "initial_ch": env.ch,
                    "initial_members": env.member_count,
                    "input_dim": int(observation.shape[1]),
                    "schedule_coverage_rounds": len(bundle["schedule"]),
                    "schedule_cache": schedule_metadata["cache_file"],
                }
            )
    if not environments:
        raise RuntimeError("dynamic curriculum contains no target ranks")
    return environments, curriculum_manifest, cfg


def padded_state(env, observation, mask, max_branches):
    if observation.shape[0] != max_branches:
        raise ValueError("dynamic observation lost global node identity")
    caps = np.minimum(env.base.queue, env.base.cfg.n_max).astype(np.int64)
    caps[~mask] = 0
    return observation.astype(np.float32), mask.astype(bool), caps


def greedy_curriculum_evaluation(agent, environments, max_branches, reward_model):
    rows = []
    first_observation = None
    first_env = None
    for env in environments:
        observation, mask, _ = env.reset()
        if first_observation is None:
            first_observation = observation.copy()
            first_env = env
        padded, padded_mask, caps = padded_state(
            env, observation, mask, max_branches
        )
        reward_total = 0.0
        packets = 0
        allocated = 0
        zero_steps = 0
        steps = 0
        done = False
        while not done:
            action, _ = agent.act(
                padded, padded_mask, epsilon=0.0, caps=caps
            )
            actual_action = action
            next_observation, next_mask, done, info = env.step(actual_action)
            reward, _ = reward_model.evaluate(info["reward_raw_terms"])
            reward_total += reward
            packets += int(info["target_packets_delivered"])
            allocated += int(actual_action.sum())
            zero_steps += int(actual_action.sum() == 0)
            steps += 1
            padded, padded_mask, caps = padded_state(
                env, next_observation, next_mask, max_branches
            )
        rows.append(
            {
                "seed": env.seed,
                "target_rank": env.target_rank,
                "cluster": info["target_cluster"],
                "members": len(info["target_members"]),
                "reward": reward_total,
                "packets": packets,
                "steps": steps,
                "packets_per_step": packets / max(1, steps),
                "mean_allocated_slots": allocated / max(1, steps),
                "zero_action_fraction": zero_steps / max(1, steps),
                "target_ch_alive": bool(env.base.alive[int(info["target_ch"])]),
            }
        )
    return {
        "clusters": rows,
        "mean_reward": float(np.mean([row["reward"] for row in rows])),
        "mean_packets": float(np.mean([row["packets"] for row in rows])),
        "mean_packets_per_step": float(
            np.mean([row["packets_per_step"] for row in rows])
        ),
        "mean_zero_action_fraction": float(
            np.mean([row["zero_action_fraction"] for row in rows])
        ),
    }, first_env, first_observation


def main():
    args = parse_args()
    seeds = [
        int(value)
        for value in args.development_seeds.split(",")
        if value.strip()
    ]
    if not seeds:
        raise ValueError("at least one development seed is required")
    held_out = set(range(3100, 3105))
    overlap = held_out.intersection(seeds)
    if overlap:
        raise ValueError(f"held-out pilot seeds cannot train: {sorted(overlap)}")

    set_seeds(args.optimizer_seed)
    run_dir = ROOT / "outputs" / "phase2" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = run_dir / "episodes.jsonl"
    if episodes_path.exists():
        episodes_path.unlink()

    environments, curriculum_manifest, env_cfg = build_curriculum(
        seeds, args.max_steps
    )
    max_branches = environments[0].base.n_nodes
    input_dims = {entry["input_dim"] for entry in curriculum_manifest}
    if len(input_dims) != 1:
        raise RuntimeError(f"inconsistent input dimensions: {input_dims}")
    input_dim = input_dims.pop()
    _, reward_payload = load_reward_model()
    reward_weights = dict(reward_payload["weights"])
    if args.idle_weight is not None:
        reward_weights["idle_energy_j"] = args.idle_weight
    if args.death_weight is not None:
        reward_weights["deaths"] = args.death_weight
    reward_payload = dict(reward_payload)
    reward_payload["weights"] = reward_weights
    reward_model = RewardModel(reward_payload["scales"], reward_weights)
    agent_cfg = BranchingAgentConfig(
        input_dim=input_dim,
        actions=env_cfg.n_max + 1,
        budget=(
            env_cfg.frame_slot_budget
            if args.projection_budget is None
            else args.projection_budget
        ),
        replay_capacity=5000,
    )
    agent = BranchingDQNAgent(agent_cfg, device=args.device)
    initialization = None
    if args.initial_checkpoint:
        checkpoint_path = Path(args.initial_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT / checkpoint_path
        checkpoint = torch.load(
            checkpoint_path, map_location=args.device, weights_only=False
        )
        agent.online.load_state_dict(checkpoint["online_state_dict"] )
        agent.target.load_state_dict(checkpoint["target_state_dict"] )
        initialization = str(checkpoint_path)

    expected_steps = args.episodes * args.max_steps
    rows = []
    recent = deque(maxlen=args.collapse_window)
    global_step = 0
    stopped_for_collapse = False
    stopped_for_nonfinite = False
    order_rng = np.random.default_rng(args.optimizer_seed)
    order = np.arange(len(environments))

    for episode in range(args.episodes):
        offset = episode % len(environments)
        if offset == 0:
            order_rng.shuffle(order)
        env = environments[int(order[offset])]
        observation, mask, _ = env.reset()
        padded, padded_mask, caps = padded_state(
            env, observation, mask, max_branches
        )
        epsilon = epsilon_for(
            episode, args.episodes, args.epsilon_start, args.epsilon_end
        )
        episode_reward = 0.0
        raw_sums = {name: 0.0 for name in TERM_ORDER}
        weighted_sums = {name: 0.0 for name in TERM_ORDER}
        losses = []
        packets = 0
        zero_action_steps = 0
        allocated_slots = 0
        done = False
        steps = 0

        while not done:
            action, _ = agent.act(
                padded, padded_mask, epsilon=epsilon, caps=caps
            )
            actual_action = action
            next_observation, next_mask, done, info = env.step(actual_action)
            next_padded, next_padded_mask, next_caps = padded_state(
                env, next_observation, next_mask, max_branches
            )
            reward, weighted = reward_model.evaluate(info["reward_raw_terms"])
            if not np.isfinite(reward) or not np.all(np.isfinite(next_padded)):
                stopped_for_nonfinite = True
                done = True
                break
            agent.store(
                padded,
                action,
                reward,
                next_padded,
                done,
                padded_mask,
                next_padded_mask,
                caps=caps,
                next_caps=next_caps,
            )
            loss = None
            if global_step % args.learn_every == 0:
                loss = agent.learn(
                    beta=beta_for(global_step, expected_steps)
                )
            if loss is not None:
                if not np.isfinite(loss):
                    stopped_for_nonfinite = True
                    done = True
                    break
                losses.append(loss)
            episode_reward += reward
            packets += int(info["target_packets_delivered"])
            zero_action_steps += int(actual_action.sum() == 0)
            allocated_slots += int(actual_action.sum())
            for name in TERM_ORDER:
                raw_sums[name] += float(info["reward_raw_terms"][name])
                weighted_sums[name] += float(weighted[name])
            padded = next_padded
            padded_mask = next_padded_mask
            caps = next_caps
            steps += 1
            global_step += 1

        row = {
            "episode": episode + 1,
            "seed": env.seed,
            "target_rank": env.target_rank,
            "target_cluster": info["target_cluster"],
            "target_ch": info["target_ch"],
            "target_members": len(info["target_members"]),
            "epsilon": epsilon,
            "steps": steps,
            "reward": episode_reward,
            "target_packets": packets,
            "packets_per_step": packets / max(1, steps),
            "zero_action_fraction": zero_action_steps / max(1, steps),
            "mean_allocated_slots": allocated_slots / max(1, steps),
            "mean_loss": float(np.mean(losses)) if losses else None,
            "raw_terms": raw_sums,
            "weighted_terms": weighted_sums,
            "t_fnd": env.base.t_fnd,
            "target_ch_alive": bool(env.base.alive[int(info["target_ch"])]),
            "dropped_stale_packets": env.base.dropped_stale_packets,
            "dropped_overflow_packets": env.base.dropped_overflow_packets,
        }
        rows.append(row)
        recent.append(row)
        with episodes_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

        if (episode + 1) % 10 == 0 or episode == 0:
            print(
                f"EPISODE={episode + 1}/{args.episodes} "
                f"SEED={env.seed} RANK={env.target_rank} "
                f"REWARD={episode_reward:.4f} PACKETS={packets} "
                f"ZERO={row['zero_action_fraction']:.3f} "
                f"LOSS={row['mean_loss']} EPS={epsilon:.3f}",
                flush=True,
            )

        if stopped_for_nonfinite:
            break
        if len(recent) == args.collapse_window and episode + 1 >= 100:
            zero_fraction = float(
                np.mean([item["zero_action_fraction"] for item in recent])
            )
            packet_rate = float(
                np.mean([item["packets_per_step"] for item in recent])
            )
            if zero_fraction > 0.80 and packet_rate < 0.10:
                stopped_for_collapse = True
                print(
                    f"ALWAYS_SLEEP_COLLAPSE=TRUE ZERO={zero_fraction:.4f} "
                    f"PACKETS_PER_STEP={packet_rate:.4f}",
                    flush=True,
                )
                break

    evaluation, check_env, check_observation = greedy_curriculum_evaluation(
        agent, environments, max_branches, reward_model
    )
    q_check = trajectory_q_check(agent, check_env, check_observation)
    balance_rows = rows[-min(50, len(rows)) :]
    contribution_totals, contribution_fractions, dominant = contribution_balance(
        balance_rows
    )
    dominating = bool(
        dominant is not None and contribution_fractions[dominant] > 0.80
    )
    collapse = bool(
        stopped_for_collapse
        or evaluation["mean_zero_action_fraction"] > 0.80
        or evaluation["mean_packets"] <= 0.0
    )

    convergence = {"assessed": len(rows) >= 100, "relative_change": None, "pass": False}
    if len(rows) >= 100:
        previous = np.mean([row["reward"] for row in rows[-100:-50]])
        current = np.mean([row["reward"] for row in rows[-50:]])
        relative = abs(current - previous) / max(1.0, abs(previous))
        convergence = {
            "assessed": True,
            "previous_50_mean_reward": float(previous),
            "last_50_mean_reward": float(current),
            "relative_change": float(relative),
            "pass": bool(relative <= 0.10),
        }

    visited = {
        (int(row["seed"]), int(row["target_rank"])) for row in rows
    }
    full_curriculum_seen = len(visited) == len(environments)
    smoke_only = args.episodes < 500
    gate_pass = bool(
        not smoke_only
        and len(rows) == args.episodes
        and full_curriculum_seen
        and not stopped_for_nonfinite
        and not collapse
        and not dominating
        and q_check["differentiated"]
        and convergence["pass"]
    )
    status = (
        "smoke_pass"
        if smoke_only and not stopped_for_nonfinite and not collapse
        else "pass" if gate_pass else "fail"
    )
    summary = {
        "status": status,
        "phase2_curriculum_gate_pass": gate_pass,
        "smoke_only": smoke_only,
        "episodes_requested": args.episodes,
        "episodes_completed": len(rows),
        "global_steps": global_step,
        "optimizer_seed": args.optimizer_seed,
        "learn_every_environment_steps": args.learn_every,
        "initial_checkpoint": initialization,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "projection_budget": agent_cfg.budget,
        "environment_capacity": env_cfg.frame_slot_budget,
        "development_seeds": seeds,
        "held_out_pilot_seeds": sorted(held_out),
        "held_out_overlap": sorted(overlap),
        "git_hash": git_hash(),
        "curriculum_clusters": curriculum_manifest,
        "curriculum_pair_count": len(environments),
        "full_curriculum_seen": full_curriculum_seen,
        "max_padded_branches": max_branches,
        "agent_config": agent_cfg.__dict__,
        "reward_calibration": reward_payload,
        "queue_feasibility_caps": True,
        "target_ch_death_penalized": True,
        "always_sleep_collapse": collapse,
        "nonfinite_detected": stopped_for_nonfinite,
        "reward_balance": {
            "last_n_episodes": len(balance_rows),
            "absolute_totals": contribution_totals,
            "fractions": contribution_fractions,
            "dominant_term": dominant,
            "pathological_domination": dominating,
            "threshold": 0.80,
        },
        "greedy_evaluation": evaluation,
        "trajectory_q_check": q_check,
        "convergence": convergence,
        "scope": (
            "Development curriculum uses frozen per-round CH schedule replay; "
            "held-out Phase 3 seeds are excluded and CH selection remains exogenous."
        ),
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    checkpoint_path = run_dir / "branching_c51.pt"
    agent.save(checkpoint_path, summary)

    print(f"CURRICULUM_PAIRS={len(environments)}")
    print(f"MAX_PADDED_BRANCHES={max_branches}")
    print(f"EPISODES_COMPLETED={len(rows)}")
    print(f"FULL_CURRICULUM_SEEN={full_curriculum_seen}")
    print(f"ALWAYS_SLEEP_COLLAPSE={collapse}")
    print(f"REWARD_PATHOLOGICAL_DOMINATION={dominating}")
    print(f"GREEDY_MEAN_PACKETS={evaluation['mean_packets']:.4f}")
    print(f"S8_S1_Q_MAX_ABS_DIFF={q_check['max_absolute_difference']:.8f}")
    print(f"CONVERGENCE_PASS={convergence['pass']}")
    print(f"PHASE2_CURRICULUM_GATE_PASS={gate_pass}")
    print(f"summary={summary_path}")
    return 0 if status in {"smoke_pass", "pass"} else 2


if __name__ == "__main__":
    raise SystemExit(main())