"""Recover Phase 2 finalization from an archived episode-tail checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.branching_dqn import BranchingAgentConfig, BranchingDQNAgent
from agents.reward_model import RewardModel
from core.reproducibility import sha256_file
from experiments.train_phase2_dynamic_curriculum import (
    build_curriculum,
    greedy_curriculum_evaluation,
    policy_stability_summary,
    reset_inspection_state,
)
from experiments.train_phase2_fixed_cluster import (
    contribution_balance,
    git_hash,
    load_reward_model,
    trajectory_q_check,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--optimizer-seed", type=int, required=True)
    parser.add_argument("--training-git-hash", required=True)
    parser.add_argument(
        "--development-seeds", default="2300,2301,2302,2303,2304"
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--stability-relative-tolerance", type=float, default=0.10)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def finite_nested(value):
    if isinstance(value, dict):
        return all(finite_nested(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_nested(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(np.isfinite(float(value)))
    return True


def main():
    args = parse_args()
    seeds = [int(value) for value in args.development_seeds.split(",") if value]
    run_dir = ROOT / "outputs" / "phase2" / args.run_name
    episodes_path = run_dir / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(episodes_path)
    rows = [
        json.loads(line)
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != args.episodes or int(rows[-1]["episode"]) != args.episodes:
        raise RuntimeError(
            f"cannot recover incomplete training: rows={len(rows)}, "
            f"last_episode={rows[-1].get('episode') if rows else None}"
        )

    source_path = run_dir / f"stability_episode_{args.episodes}.pt"
    source = torch.load(source_path, map_location=args.device, weights_only=False)
    config = BranchingAgentConfig(**source["config"])
    agent = BranchingDQNAgent(config, device=args.device)
    agent.online.load_state_dict(source["online_state_dict"])
    agent.target.load_state_dict(source["target_state_dict"])
    agent.optimizer.load_state_dict(source["optimizer_state_dict"])

    environments, curriculum_manifest, env_cfg = build_curriculum(
        seeds, args.max_steps
    )
    max_branches = environments[0].base.n_nodes
    if config.max_branches != max_branches:
        raise RuntimeError("checkpoint branch count does not match curriculum")
    _, reward_payload = load_reward_model()
    reward_model = RewardModel(
        reward_payload["scales"], reward_payload["weights"]
    )

    stability_snapshots = []
    for episode in (400, 450, 500):
        path = run_dir / f"stability_episode_{episode}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata", {})
        if int(metadata.get("episode", -1)) != episode:
            raise RuntimeError(f"invalid stability metadata in {path}")
        stability_snapshots.append(metadata)

    evaluation, check_env, _ = greedy_curriculum_evaluation(
        agent, environments, max_branches, reward_model
    )
    check_observation = reset_inspection_state(check_env)
    q_check = trajectory_q_check(agent, check_env, check_observation)
    policy_stability = policy_stability_summary(
        stability_snapshots, args.stability_relative_tolerance
    )

    balance_rows = rows[-min(50, len(rows)) :]
    contribution_totals, contribution_fractions, dominant = contribution_balance(
        balance_rows
    )
    dominating = bool(
        dominant is not None and contribution_fractions[dominant] > 0.80
    )
    nonfinite = not all(finite_nested(row) for row in rows)
    collapse = bool(
        evaluation["mean_zero_action_fraction"] > 0.80
        or evaluation["mean_packets"] <= 0.0
    )
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
    visited = {(int(row["seed"]), int(row["target_rank"])) for row in rows}
    full_curriculum_seen = len(visited) == len(environments)
    gate_pass = bool(
        full_curriculum_seen
        and not nonfinite
        and not collapse
        and not dominating
        and q_check["differentiated"]
        and convergence["pass"]
        and policy_stability["pass"]
    )

    summary = {
        "status": "pass" if gate_pass else "fail",
        "phase2_curriculum_gate_pass": gate_pass,
        "smoke_only": False,
        "episodes_requested": args.episodes,
        "episodes_completed": len(rows),
        "global_steps": int(sum(int(row["steps"]) for row in rows)),
        "optimizer_seed": args.optimizer_seed,
        "learn_every_environment_steps": 4,
        "initial_checkpoint": None,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "projection_budget": config.budget,
        "environment_capacity": env_cfg.frame_slot_budget,
        "development_seeds": seeds,
        "schedule_schema_version": curriculum_manifest[0]["schedule_schema_version"],
        "held_out_pilot_seeds": list(range(3100, 3105)),
        "held_out_overlap": [],
        "git_hash": args.training_git_hash,
        "finalization_git_hash": git_hash(),
        "curriculum_clusters": curriculum_manifest,
        "curriculum_pair_count": len(environments),
        "full_curriculum_seen": full_curriculum_seen,
        "max_padded_branches": max_branches,
        "agent_config": config.__dict__,
        "online_parameter_count": int(
            sum(parameter.numel() for parameter in agent.online.parameters())
        ),
        "reward_calibration": reward_payload,
        "queue_feasibility_caps": True,
        "target_ch_death_penalized": True,
        "always_sleep_collapse": collapse,
        "nonfinite_detected": nonfinite,
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
        "policy_stability": policy_stability,
        "policy_stability_snapshots": stability_snapshots,
        "recovery": {
            "reason": "post_training_terminal_inspection_state_error",
            "weights_retrained": False,
            "source_checkpoint": str(source_path.relative_to(ROOT)).replace("\\", "/"),
            "source_checkpoint_sha256": sha256_file(source_path),
        },
        "scope": (
            "Development curriculum uses frozen per-round CH schedule replay; "
            "held-out Phase 3 seeds are excluded and CH selection remains exogenous."
        ),
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    checkpoint_path = run_dir / "branching_c51.pt"
    agent.save(checkpoint_path, summary)

    print(f"RECOVERED_EPISODES={len(rows)}")
    print("WEIGHTS_RETRAINED=False")
    print(f"S8_S1_Q_MAX_ABS_DIFF={q_check['max_absolute_difference']:.8f}")
    print(f"CONVERGENCE_PASS={convergence['pass']}")
    print(f"POLICY_STABILITY_PASS={policy_stability['pass']}")
    print(f"PHASE2_CURRICULUM_GATE_PASS={gate_pass}")
    print(f"summary={summary_path}")
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())