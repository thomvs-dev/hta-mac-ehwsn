"""Archive the fixed-cluster controls used to inspect the Phase 2 gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.branching_dqn import BranchingAgentConfig, BranchingDQNAgent
from agents.budget_projection import project_slot_budget
from experiments.train_phase2_fixed_cluster import (
    ROOT,
    build_environment,
    load_reward_model,
    set_seeds,
)


def run_policy(name, env, cfg, reward_model, agent, rng):
    observation, mask, _ = env.reset()
    done = False
    total_reward = 0.0
    packets = 0
    slots = 0
    steps = 0
    raw = {key: 0.0 for key in reward_model.weights}
    action_histogram = np.zeros(cfg.n_max + 1, dtype=np.int64)
    while not done:
        if name == "always_sleep":
            action = np.zeros(env.member_count, dtype=np.int64)
        elif name == "static_equal":
            action = np.zeros(env.member_count, dtype=np.int64)
            action[np.flatnonzero(mask)[: cfg.frame_slot_budget]] = 1
        elif name == "random_budgeted":
            q_values = rng.normal(size=(env.member_count, cfg.n_max + 1))
            q_values[:, 1:] += 0.5
            action = project_slot_budget(
                q_values,
                cfg.frame_slot_budget,
                stop_at_nonpositive_gain=False,
            )
            action[~mask] = 0
        else:
            action, _ = agent.act(observation, mask, epsilon=0.0)
        action_histogram += np.bincount(action, minlength=cfg.n_max + 1)
        observation, mask, done, info = env.step(action)
        reward, _ = reward_model.evaluate(info["reward_raw_terms"])
        total_reward += reward
        packets += int(info["target_packets_delivered"])
        slots += int(action.sum())
        steps += 1
        for key, value in info["reward_raw_terms"].items():
            raw[key] += float(value)
    return {
        "policy": name,
        "reward": total_reward,
        "packets": packets,
        "steps": steps,
        "mean_slots": slots / steps,
        "action_histogram": action_histogram.tolist(),
        "raw_reward_terms": raw,
    }


def main():
    set_seeds(2100)
    env, cfg = build_environment(2100, 150)
    reward_model, _ = load_reward_model()
    observation, _, _ = env.reset()
    agent = BranchingDQNAgent(
        BranchingAgentConfig(
            input_dim=observation.shape[1],
            actions=cfg.n_max + 1,
            budget=cfg.frame_slot_budget,
        )
    )
    checkpoint_path = (
        ROOT
        / "outputs"
        / "phase2"
        / "authoritative_500ep_seed2100"
        / "branching_c51.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    agent.online.load_state_dict(checkpoint["online_state_dict"])
    agent.target.load_state_dict(checkpoint["target_state_dict"])
    rng = np.random.default_rng(991)
    rows = [
        run_policy(name, env, cfg, reward_model, agent, rng)
        for name in (
            "always_sleep",
            "static_equal",
            "random_budgeted",
            "trained_greedy",
        )
    ]
    indexed = {row["policy"]: row for row in rows}
    learned = indexed["trained_greedy"]
    static = indexed["static_equal"]
    random_policy = indexed["random_budgeted"]
    report = {
        "status": "phase2_control_inspection_complete",
        "scope": "one deterministic fixed seed-2100 cluster; not a multi-trial superiority result",
        "seed": 2100,
        "target_members": env.member_count,
        "policies": rows,
        "trained_minus_static": {
            "reward": learned["reward"] - static["reward"],
            "packets": learned["packets"] - static["packets"],
            "steps": learned["steps"] - static["steps"],
            "idle_energy_j": (
                learned["raw_reward_terms"]["idle_energy_j"]
                - static["raw_reward_terms"]["idle_energy_j"]
            ),
        },
        "trained_minus_random": {
            "reward": learned["reward"] - random_policy["reward"],
            "packets": learned["packets"] - random_policy["packets"],
            "steps": learned["steps"] - random_policy["steps"],
        },
        "superiority_established": False,
        "reason": (
            "The trained policy exceeds static reward in this one fixed episode "
            "but not the random-budgeted control; Phase 3/4 paired trials remain required."
        ),
    }
    output = ROOT / "outputs" / "logs" / "phase2_fixed_cluster_controls.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"TRAINED_REWARD={learned['reward']:.8f}")
    print(f"STATIC_REWARD={static['reward']:.8f}")
    print(f"RANDOM_REWARD={random_policy['reward']:.8f}")
    print(f"TRAINED_STEPS={learned['steps']}")
    print(f"STATIC_STEPS={static['steps']}")
    print("SUPERIORITY_ESTABLISHED=False")
    print(f"report={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
