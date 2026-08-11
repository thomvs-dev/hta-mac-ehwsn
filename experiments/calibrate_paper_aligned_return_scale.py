"""Calibrate C51 reward scale on paper-aligned development rollouts only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



from agents.qos_constraints import QoSConstraintController
from agents.reward_model import RewardModel
from envs.policy_observation import PHASE2D_POLICY_SCHEMA
from experiments.train_phase2_dynamic_curriculum import (
    ROOT,
    build_curriculum,
    load_qos_constraints,
    load_reward_model,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-profile", required=True)
    parser.add_argument("--qos-constraint-config", required=True)
    parser.add_argument("--development-seeds", default="2400,2401,2402,2403,2404")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--rollouts", type=int, default=100)
    parser.add_argument("--audit-seed", type=int, default=20260806)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--quantile-probability", type=float, default=0.995)
    parser.add_argument("--headroom-fraction", type=float, default=0.8)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def random_budgeted_action(rng, caps, mask, budget):
    caps = np.asarray(caps, dtype=np.int64)
    mask = np.asarray(mask, dtype=bool)
    desired = np.zeros(len(caps), dtype=np.int64)
    active = np.flatnonzero(mask)
    if active.size:
        desired[active] = np.asarray(
            [rng.integers(0, int(caps[node]) + 1) for node in active],
            dtype=np.int64,
        )
    while int(desired.sum()) > int(budget):
        candidates = np.flatnonzero(desired > 0)
        desired[int(rng.choice(candidates))] -= 1
    return desired


def main():
    args = parse_args()
    if not 0.5 < args.quantile_probability < 1.0:
        raise ValueError("quantile probability must lie in (0.5, 1)")
    if not 0.0 < args.headroom_fraction < 1.0:
        raise ValueError("headroom fraction must lie in (0, 1)")
    seeds = [int(value) for value in args.development_seeds.split(",") if value.strip()]
    if set(seeds).intersection(range(3100, 3105)):
        raise ValueError("registered held-out seeds cannot calibrate reward scale")
    environments, manifest, config = build_curriculum(
        seeds,
        args.max_steps,
        observation_schema=PHASE2D_POLICY_SCHEMA,
        environment_profile=args.environment_profile,
    )
    qos_config, qos_evidence = load_qos_constraints(args.qos_constraint_config)
    controller = QoSConstraintController(qos_config)
    _, reward_payload = load_reward_model()
    reward_model = RewardModel(reward_payload["scales"], reward_payload["weights"])
    rng = np.random.default_rng(args.audit_seed)
    returns = []
    episode_lengths = []
    for rollout in range(args.rollouts):
        env = environments[rollout % len(environments)]
        _, mask, _ = env.reset()
        controller.begin_episode()
        rewards = []
        done = False
        while not done:
            caps = np.minimum(env.base.queue, env.base.cfg.n_max).astype(np.int64)
            caps[~mask] = 0
            action = random_budgeted_action(rng, caps, mask, config.frame_slot_budget)
            _, mask, done, info = env.step(action)
            physical, _ = reward_model.evaluate(info["reward_raw_terms"])
            penalty, _ = controller.evaluate_info(info)
            rewards.append(float(physical + penalty))
        controller.end_episode()
        episode_lengths.append(len(rewards))
        value = 0.0
        episode_returns = []
        for reward in reversed(rewards):
            value = reward + args.gamma * value
            episode_returns.append(value)
        returns.extend(reversed(episode_returns))
    values = np.asarray(returns, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise RuntimeError("calibration produced no finite returns")
    lower_probability = 1.0 - args.quantile_probability
    q_low = float(np.quantile(values, lower_probability))
    q_high = float(np.quantile(values, args.quantile_probability))
    q_star = max(abs(q_low), abs(q_high), 1e-12)
    support = {"v_min": -30.0, "v_max": 30.0, "atoms": 51}
    scale = min(1.0, args.headroom_fraction * support["v_max"] / q_star)
    profile_path = Path(args.environment_profile)
    if not profile_path.is_absolute():
        profile_path = ROOT / profile_path
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "status": "frozen_development_scale",
        "reward_scale": float(scale),
        "apply_to": "replay_and_c51_reward_only",
        "physical_metrics_scaled": False,
        "calibration_method": "random_budgeted_discounted_return_two_sided_quantiles",
        "q_low": q_low,
        "q_high": q_high,
        "q_star": q_star,
        "return_count": int(values.size),
        "rollouts": int(args.rollouts),
        "episode_length_min": int(min(episode_lengths)),
        "episode_length_max": int(max(episode_lengths)),
        "gamma": float(args.gamma),
        "support": support,
        "headroom_fraction": float(args.headroom_fraction),
        "quantile_probability": float(args.quantile_probability),
        "development_seeds": seeds,
        "environment_profile_sha256": file_sha256(profile_path),
        "qos_constraint_sha256": qos_evidence["sha256"],
        "schedule_schema_version": manifest[0]["schedule_schema_version"],
        "held_out_seeds_used": False,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"RETURN_COUNT={values.size}")
    print(f"Q_LOW={q_low:.12f}")
    print(f"Q_HIGH={q_high:.12f}")
    print(f"REWARD_SCALE={scale:.12f}")
    print(f"OUTPUT={output}")


if __name__ == "__main__":
    main()
