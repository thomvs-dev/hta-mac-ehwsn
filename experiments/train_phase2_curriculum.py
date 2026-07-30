"""Train HTA-MAC across multiple frozen development clusters with padding."""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.branching_dqn import BranchingAgentConfig, BranchingDQNAgent
from agents.reward_model import TERM_ORDER
from envs import IntraClusterMACEnv
from envs.fixed_cluster_training_env import FixedClusterTrainingEnv
from experiments.train_phase2_fixed_cluster import (
    beta_for,
    build_environment,
    contribution_balance,
    git_hash,
    load_reward_model,
    set_seeds,
    trajectory_q_check,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument(
        "--development-seeds",
        default="2200,2201,2202,2203,2204,2205,2206,2207,2208,2209",
    )
    parser.add_argument("--optimizer-seed", type=int, default=2199)
    parser.add_argument("--run-name", default="curriculum_600ep_dev10")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--collapse-window", type=int, default=50)
    return parser.parse_args()


def epsilon_for(episode: int, total_episodes: int) -> float:
    anneal_episodes = max(1, int(total_episodes * 0.8))
    fraction = min(1.0, episode / anneal_episodes)
    return 1.0 + fraction * (0.05 - 1.0)


def clone_target_environment(prototype, target_cluster: int):
    base = prototype.base
    cloned_base = IntraClusterMACEnv(
        base.cfg,
        base.radio,
        base.solar,
        base.thermal,
        idle_energy_enabled=base.idle_energy_enabled,
    )
    return FixedClusterTrainingEnv(
        cloned_base,
        prototype.snapshot,
        seed=prototype.seed,
        target_cluster=target_cluster,
    )


def build_curriculum(seeds, max_steps):
    environments = []
    manifest = []
    for seed in seeds:
        prototype, cfg = build_environment(seed, max_steps)
        cluster_count = len(prototype.snapshot["cluster_heads"])
        for cluster in range(cluster_count):
            env = clone_target_environment(prototype, cluster)
            try:
                observation, _, _ = env.reset()
            except ValueError as exc:
                if "no member branches" in str(exc):
                    continue
                raise
            environments.append(env)
            manifest.append(
                {
                    "seed": seed,
                    "cluster": cluster,
                    "ch": env.ch,
                    "members": env.member_count,
                    "input_dim": int(observation.shape[1]),
                }
            )
    if not environments:
        raise RuntimeError("curriculum contains no viable clusters")
    return environments, manifest, cfg


def padded_state(env, observation, mask, max_branches):
    padded = np.zeros((max_branches, observation.shape[1]), dtype=np.float32)
    padded_mask = np.zeros(max_branches, dtype=bool)
    caps = np.zeros(max_branches, dtype=np.int64)
    count = env.member_count
    padded[:count] = observation
    padded_mask[:count] = mask
    caps[:count] = np.minimum(
        env.base.queue[env.members], env.base.cfg.n_max
    )
    caps[~padded_mask] = 0
    return padded, padded_mask, caps


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
            actual_action = action[: env.member_count]
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
                "cluster": env.target_cluster,
                "members": env.member_count,
                "reward": reward_total,
                "packets": packets,
                "steps": steps,
                "packets_per_step": packets / max(1, steps),
                "mean_allocated_slots": allocated / max(1, steps),
                "zero_action_fraction": zero_steps / max(1, steps),
                "target_ch_alive": bool(env.base.alive[env.ch]),
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
    max_branches = max(env.member_count for env in environments)
    input_dims = {entry["input_dim"] for entry in curriculum_manifest}
    if len(input_dims) != 1:
        raise RuntimeError(f"inconsistent input dimensions: {input_dims}")
    input_dim = input_dims.pop()
    reward_model, reward_payload = load_reward_model()
    agent_cfg = BranchingAgentConfig(
        input_dim=input_dim,
        actions=env_cfg.n_max + 1,
        budget=env_cfg.frame_slot_budget,
    )
    agent = BranchingDQNAgent(agent_cfg, device=args.device)

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
        epsilon = epsilon_for(episode, args.episodes)
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
            actual_action = action[: env.member_count]
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
            loss = agent.learn(beta=beta_for(global_step, expected_steps))
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
            "target_cluster": env.target_cluster,
            "target_ch": env.ch,
            "target_members": env.member_count,
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
            "target_ch_alive": bool(env.base.alive[env.ch]),
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
                f"SEED={env.seed} CLUSTER={env.target_cluster} "
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
        (int(row["seed"]), int(row["target_cluster"])) for row in rows
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
            "Balanced development curriculum over every frozen initial cluster; "
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