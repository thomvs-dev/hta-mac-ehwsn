"""Train and inspect HTA-MAC on the Phase 2 fixed-cluster gate."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
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
from core.ch_selection.frozen_heart_ch import FrozenHeartCH
from core.ch_selection.initial_snapshot import frozen_initial_snapshot
from core.configuration import load_simple_yaml
from core.energy.radio_model import RadioModel
from core.hmm import load_solar_hmm, load_thermal_auxiliary
from core.hmm.rectified_moments import next_rectified_statistics
from envs import IntraClusterMACEnv, MACEnvironmentConfig
from envs.fixed_cluster_training_env import FixedClusterTrainingEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=2100)
    parser.add_argument("--run-name", default="phase2_fixed_cluster_500ep")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--collapse-window", type=int, default=50)
    return parser.parse_args()


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def build_environment(seed: int, max_steps: int):
    base = load_simple_yaml(ROOT / "config" / "base.yaml")
    mac = load_simple_yaml(ROOT / "config" / "phase1.yaml")
    manifest = load_simple_yaml(ROOT / "core" / "frozen_assets.yaml")
    upstream = ROOT.parent / "final_repo"
    policy = FrozenHeartCH(
        upstream,
        manifest["checkpoint"]["path"],
        manifest["checkpoint"]["sha256"],
    )
    snapshot = frozen_initial_snapshot(policy, seed)
    solar = load_solar_hmm(upstream / manifest["solar_hmm"]["path"])
    thermal = load_thermal_auxiliary(
        ROOT / manifest["thermal_hmm"]["auxiliary_path"]
    )
    radio = RadioModel(
        e_elec_j_per_bit=base["radio"]["e_elec_j_per_bit"],
        eps_fs_j_per_bit_m2=base["radio"]["eps_fs_j_per_bit_m2"],
        eps_mp_j_per_bit_m4=base["radio"]["eps_mp_j_per_bit_m4"],
        e_da_j_per_bit=base["radio"]["e_da_j_per_bit"],
        d0_m=base["radio"]["d0_m"],
    )
    cfg = MACEnvironmentConfig(
        initial_energy_j=base["network"]["initial_energy_j"],
        packet_bits=base["network"]["packet_bits"],
        control_packet_bits=base["network"]["control_packet_bits"],
        e_elec_j_per_bit=base["radio"]["e_elec_j_per_bit"],
        frame_slot_budget=mac["frame_slot_budget"],
        n_max=mac["n_max"],
        queue_max_packets=mac["queue_max_packets"],
        packet_ttl_rounds=mac["packet_ttl_rounds"],
        max_rounds=max_steps,
        solar_scale=base["harvesting"]["solar"]["rectification_scale"],
        thermal_scale=base["harvesting"]["thermal"]["rectification_scale"],
        bs_position_m=tuple(base["network"]["bs_position_m"]),
        idle_slot_bit_times=mac["idle_energy"]["primary_slot_bit_times"],
    )
    base_env = IntraClusterMACEnv(cfg, radio, solar, thermal)
    return FixedClusterTrainingEnv(base_env, snapshot, seed=seed), cfg


def load_reward_model():
    payload = json.loads(
        (ROOT / "config" / "reward_calibration.json").read_text(encoding="utf-8")
    )
    return RewardModel(payload["scales"], payload["weights"]), payload


def epsilon_for(episode: int):
    fraction = min(1.0, episode / 400.0)
    return 1.0 + fraction * (0.05 - 1.0)


def beta_for(step: int, expected_steps: int):
    return min(1.0, 0.4 + 0.6 * step / max(1, expected_steps))


def contribution_balance(rows):
    totals = {name: 0.0 for name in TERM_ORDER}
    for row in rows:
        for name in TERM_ORDER:
            totals[name] += abs(float(row["weighted_terms"][name]))
    active_total = sum(totals.values())
    fractions = {
        name: (value / active_total if active_total else 0.0)
        for name, value in totals.items()
    }
    active = {name: value for name, value in fractions.items() if totals[name] > 0.0}
    dominant = max(active, key=active.get) if active else None
    return totals, fractions, dominant


def trajectory_q_check(agent, env, observation):
    base = observation[0].copy()
    low = base.copy()
    high = base.copy()
    low[0] = high[0] = 0.5
    low[3:11] = env.base.solar.transition[0]
    high[3:11] = env.base.solar.transition[7]
    solar_mean, solar_var = next_rectified_statistics(
        env.base.solar.transition,
        env.base.solar.mean,
        env.base.solar.variance,
        env.base.cfg.solar_scale,
    )
    low[1], low[2] = solar_mean[0], solar_var[0]
    high[1], high[2] = solar_mean[7], solar_var[7]
    pair = np.stack((low, high)).astype(np.float32)[None, :, :]
    with torch.no_grad():
        q = agent.online.q_values(
            torch.as_tensor(pair, dtype=torch.float32, device=agent.device)
        )[0].cpu().numpy()
    return {
        "equal_normalized_energy": 0.5,
        "s1_q_values": q[0].tolist(),
        "s8_q_values": q[1].tolist(),
        "max_absolute_difference": float(np.max(np.abs(q[1] - q[0]))),
        "differentiated": bool(np.max(np.abs(q[1] - q[0])) > 1e-4),
    }


def greedy_evaluation(agent, env, reward_model, episodes=10):
    rows = []
    first_observation = None
    for _ in range(episodes):
        observation, mask, _ = env.reset()
        if first_observation is None:
            first_observation = observation.copy()
        total_reward = 0.0
        packets = 0
        zero_steps = 0
        steps = 0
        done = False
        while not done:
            caps = np.minimum(
                env.base.queue[env.members], env.base.cfg.n_max
            )
            action, _ = agent.act(
                observation, mask, epsilon=0.0, caps=caps
            )
            next_observation, next_mask, done, info = env.step(action)
            reward, _ = reward_model.evaluate(info["reward_raw_terms"])
            total_reward += reward
            packets += info["target_packets_delivered"]
            zero_steps += int(action.sum() == 0)
            steps += 1
            observation, mask = next_observation, next_mask
        rows.append(
            {
                "reward": total_reward,
                "packets": packets,
                "steps": steps,
                "zero_action_fraction": zero_steps / max(1, steps),
            }
        )
    signatures = {json.dumps(row, sort_keys=True) for row in rows}
    return {
        "episodes": rows,
        "deterministic": len(signatures) == 1,
        "mean_reward": float(np.mean([row["reward"] for row in rows])),
        "mean_packets": float(np.mean([row["packets"] for row in rows])),
        "mean_zero_action_fraction": float(
            np.mean([row["zero_action_fraction"] for row in rows])
        ),
    }, first_observation


def main():
    args = parse_args()
    set_seeds(args.seed)
    run_dir = ROOT / "outputs" / "phase2" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = run_dir / "episodes.jsonl"
    if episodes_path.exists():
        episodes_path.unlink()

    env, env_cfg = build_environment(args.seed, args.max_steps)
    reward_model, reward_payload = load_reward_model()
    observation, mask, reset_info = env.reset()
    if env.member_count != 20:
        raise RuntimeError(
            f"canonical Phase 2 cluster must have 20 members, got {env.member_count}"
        )
    agent_cfg = BranchingAgentConfig(
        input_dim=observation.shape[1],
        actions=env_cfg.n_max + 1,
        budget=env_cfg.frame_slot_budget,
    )
    agent = BranchingDQNAgent(agent_cfg, device=args.device)
    expected_steps = args.episodes * args.max_steps
    recent = deque(maxlen=args.collapse_window)
    rows = []
    global_step = 0
    stopped_for_collapse = False
    stopped_for_nonfinite = False

    for episode in range(args.episodes):
        observation, mask, _ = env.reset()
        epsilon = epsilon_for(episode)
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
            caps = np.minimum(
                env.base.queue[env.members], env.base.cfg.n_max
            )
            action, _ = agent.act(
                observation, mask, epsilon=epsilon, caps=caps
            )
            next_observation, next_mask, done, info = env.step(action)
            next_caps = np.minimum(
                env.base.queue[env.members], env.base.cfg.n_max
            )
            reward, weighted = reward_model.evaluate(info["reward_raw_terms"])
            if not np.isfinite(reward) or not np.all(np.isfinite(next_observation)):
                stopped_for_nonfinite = True
                done = True
                break
            agent.store(
                observation,
                action,
                reward,
                next_observation,
                done,
                mask,
                next_mask,
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
            zero_action_steps += int(action.sum() == 0)
            allocated_slots += int(action.sum())
            for name in TERM_ORDER:
                raw_sums[name] += float(info["reward_raw_terms"][name])
                weighted_sums[name] += float(weighted[name])
            observation, mask = next_observation, next_mask
            steps += 1
            global_step += 1

        row = {
            "episode": episode + 1,
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
                f"REWARD={episode_reward:.4f} PACKETS={packets} "
                f"ZERO={row['zero_action_fraction']:.3f} "
                f"LOSS={row['mean_loss']} EPS={epsilon:.3f}",
                flush=True,
            )

        if stopped_for_nonfinite:
            break
        if len(recent) == args.collapse_window and episode + 1 >= 100:
            zero_fraction = float(np.mean([item["zero_action_fraction"] for item in recent]))
            packet_rate = float(np.mean([item["packets_per_step"] for item in recent]))
            if zero_fraction > 0.80 and packet_rate < 0.10:
                stopped_for_collapse = True
                print(
                    f"ALWAYS_SLEEP_COLLAPSE=TRUE ZERO={zero_fraction:.4f} "
                    f"PACKETS_PER_STEP={packet_rate:.4f}",
                    flush=True,
                )
                break

    evaluation, evaluation_observation = greedy_evaluation(
        agent, env, reward_model, episodes=10
    )
    q_check = trajectory_q_check(agent, env, evaluation_observation)
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

    smoke_only = args.episodes < 500
    gate_pass = bool(
        not smoke_only
        and len(rows) == 500
        and not stopped_for_nonfinite
        and not collapse
        and not dominating
        and q_check["differentiated"]
        and convergence["pass"]
    )
    status = "smoke_pass" if smoke_only and not stopped_for_nonfinite and not collapse else "pass" if gate_pass else "fail"
    summary = {
        "status": status,
        "phase2_gate_pass": gate_pass,
        "smoke_only": smoke_only,
        "episodes_requested": args.episodes,
        "episodes_completed": len(rows),
        "global_steps": global_step,
        "seed": args.seed,
        "git_hash": git_hash(),
        "target_cluster": env.target_cluster,
        "target_ch": env.ch,
        "target_members": env.member_count,
        "input_shape": [env.member_count, agent_cfg.input_dim],
        "agent_config": agent_cfg.__dict__,
        "reward_calibration": reward_payload,
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
        "fixed_cluster_scope": (
            "single immutable seed-2100 HEART-CH cluster snapshot for the Phase 2 "
            "sanity gate; full evaluation uses shared exogenous per-round replay"
        ),
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    checkpoint_path = run_dir / "branching_c51.pt"
    agent.save(checkpoint_path, summary)

    print(f"EPISODES_COMPLETED={len(rows)}")
    print(f"ALWAYS_SLEEP_COLLAPSE={collapse}")
    print(f"REWARD_PATHOLOGICAL_DOMINATION={dominating}")
    print(f"GREEDY_MEAN_PACKETS={evaluation['mean_packets']:.4f}")
    print(f"S8_S1_Q_MAX_ABS_DIFF={q_check['max_absolute_difference']:.8f}")
    print(f"CONVERGENCE_PASS={convergence['pass']}")
    print(f"PHASE2_GATE_PASS={gate_pass}")
    print(f"summary={summary_path}")
    return 0 if status in {"smoke_pass", "pass"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
