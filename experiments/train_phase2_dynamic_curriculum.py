"""Train HTA-MAC on frozen per-round CH schedules with padded branches."""

from __future__ import annotations

import argparse
import hashlib
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
from agents.qos_constraints import (
    QoSConstraintConfig,
    QoSConstraintController,
)
from agents.reward_model import RewardModel, TERM_ORDER
from core.hmm.rectified_moments import next_rectified_statistics
from envs.dynamic_cluster_training_env import DynamicClusterTrainingEnv
from envs.fixed_cluster_training_env import FixedClusterTrainingEnv
from envs.policy_observation import (
    LEGACY_POLICY_SCHEMA,
    PHASE2D_POLICY_SCHEMA,
)
from experiments.paper_aligned_environment import (
    configure_mac as configure_paper_aligned_mac,
    disabled_thermal_hmm,
    load_profile as load_environment_profile,
    schedule_bundle as paper_aligned_schedule_bundle,
)
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
    parser.add_argument("--reward-scale-config")
    parser.add_argument("--qos-constraint-config")
    parser.add_argument("--environment-profile")
    parser.add_argument(
        "--reinitialize-categorical-outputs", action="store_true"
    )
    parser.add_argument("--projection-budget", type=int)
    parser.add_argument(
        "--architecture",
        choices=(
            "shared_branching",
            "equivariant_set_branching",
            "independent_dqns",
        ),
        default="shared_branching",
    )
    parser.add_argument("--stability-interval", type=int, default=50)
    parser.add_argument("--stability-tail-episodes", type=int, default=100)
    parser.add_argument("--stability-relative-tolerance", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--normalize-input-blocks", action="store_true")
    parser.add_argument("--trajectory-loss-weight", type=float, default=0.0)
    parser.add_argument("--concavity-loss-weight", type=float, default=0.0)
    parser.add_argument("--trajectory-margin-fraction", type=float, default=0.05)
    parser.add_argument(
        "--demonstration-margin-loss-weight", type=float, default=0.0
    )
    parser.add_argument("--demonstration-margin", type=float, default=0.8)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def validate_reward_scale_payload(payload):
    if payload.get("status") != "frozen_development_scale":
        raise ValueError("reward scale is not frozen development evidence")
    if payload.get("apply_to") != "replay_and_c51_reward_only":
        raise ValueError("reward scale has an unsupported application scope")
    if payload.get("physical_metrics_scaled") is not False:
        raise ValueError("physical metrics must remain unscaled")
    if payload.get("held_out_seeds_used") is not False:
        raise ValueError("reward scale must not use held-out seeds")
    scale = float(payload["reward_scale"])
    if not np.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise ValueError("reward_scale must be finite and in (0, 1]")
    return scale


def load_reward_scale(path):
    """Load and validate the one frozen Phase 2C learning-reward scale."""
    if path is None:
        return 1.0, None
    scale_path = Path(path)
    if not scale_path.is_absolute():
        scale_path = ROOT / scale_path
    if not scale_path.is_file():
        raise FileNotFoundError(scale_path)
    payload = json.loads(scale_path.read_text(encoding="utf-8"))
    scale = validate_reward_scale_payload(payload)
    return scale, {
        "path": str(scale_path.resolve()),
        "sha256": sha256_file(scale_path),
        "payload": payload,
    }


def load_qos_constraints(path):
    """Load frozen development-only QoS thresholds and controller settings."""
    if path is None:
        return None, None
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = QoSConstraintConfig.from_payload(payload)
    evidence_path = Path(payload["feasibility_evidence"])
    if not evidence_path.is_absolute():
        evidence_path = ROOT / evidence_path
    if not evidence_path.is_file():
        raise FileNotFoundError(evidence_path)
    return config, {
        "path": str(config_path.resolve()),
        "sha256": sha256_file(config_path),
        "feasibility_evidence_path": str(evidence_path.resolve()),
        "feasibility_evidence_sha256": sha256_file(evidence_path),
        "payload": payload,
    }


def learning_reward(raw_reward, reward_scale):
    value = float(raw_reward) * float(reward_scale)
    if not np.isfinite(value):
        raise ValueError("scaled learning reward is non-finite")
    return value


def trajectory_blocks(env):
    solar_mean, solar_var = next_rectified_statistics(
        env.base.solar.transition,
        env.base.solar.mean,
        env.base.solar.variance,
        env.base.cfg.solar_scale,
    )
    thermal_mean, thermal_var = next_rectified_statistics(
        env.base.thermal.transition,
        env.base.thermal.mean,
        env.base.thermal.variance,
        env.base.cfg.thermal_scale,
    )
    low = (
        float(solar_mean[0] + thermal_mean[0]),
        float(solar_var[0] + thermal_var[0]),
        *env.base.solar.transition[0].tolist(),
        *env.base.thermal.transition[0].tolist(),
    )
    high = (
        float(solar_mean[-1] + thermal_mean[-1]),
        float(solar_var[-1] + thermal_var[-1]),
        *env.base.solar.transition[-1].tolist(),
        *env.base.thermal.transition[-1].tolist(),
    )
    harvest_scale = float(np.max(solar_mean) + np.max(thermal_mean))
    return low, high, harvest_scale


def epsilon_for(
    episode: int, total_episodes: int, start: float, end: float
) -> float:
    anneal_episodes = max(1, int(total_episodes * 0.8))
    fraction = min(1.0, episode / anneal_episodes)
    return start + fraction * (end - start)


def build_curriculum(
    seeds,
    max_steps,
    observation_schema=LEGACY_POLICY_SCHEMA,
    environment_profile=None,
):
    frozen_policy, solar, thermal, radio, cfg, manifest = build_assets(max_steps)
    profile = None
    profile_evidence = None
    idle_energy_enabled = True
    if environment_profile is not None:
        profile_path = Path(environment_profile)
        if not profile_path.is_absolute():
            profile_path = ROOT / profile_path
        profile, profile_evidence = load_environment_profile(profile_path)
        thermal = disabled_thermal_hmm(thermal)
        cfg = configure_paper_aligned_mac(profile, cfg)
        idle_energy_enabled = False
    environments = []
    curriculum_manifest = []
    checkpoint_sha = manifest["checkpoint"]["sha256"]
    for seed in seeds:
        if profile is None:
            bundle, schedule_metadata = schedule_bundle(
                frozen_policy, seed, max_steps, checkpoint_sha
            )
        else:
            bundle, schedule_metadata = paper_aligned_schedule_bundle(
                profile, solar, seed, max_steps
            )
        initial_ranks = len(bundle["schedule"][0]["cluster_heads"])
        for target_rank in range(initial_ranks):
            base = ScheduledIntraClusterMACEnv(
                cfg,
                radio,
                solar,
                thermal,
                idle_energy_enabled=idle_energy_enabled,
            )
            env = DynamicClusterTrainingEnv(
                base,
                bundle,
                seed=seed,
                target_rank=target_rank,
                observation_schema=observation_schema,
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
                    "observation_schema": env.observation_schema,
                    "observation_layout": env.observation_layout,
                    "schedule_coverage_rounds": len(bundle["schedule"]),
                    "schedule_cache": schedule_metadata.get("cache_file"),
                    "schedule_schema_version": schedule_metadata["schedule_schema_version"],
                }
            )
            if profile_evidence is not None:
                curriculum_manifest[-1]["environment_profile"] = {
                    "profile_id": profile["profile_id"],
                    "path": profile_evidence["path"],
                    "sha256": profile_evidence["sha256"],
                    "claim_boundary": profile["claim_boundary"],
                }
    if not environments:
        raise RuntimeError("dynamic curriculum contains no target ranks")
    return environments, curriculum_manifest, cfg


def padded_state(env, observation, mask, max_branches):
    if observation.shape[0] != max_branches:
        raise ValueError("dynamic observation lost global node identity")
    caps = np.minimum(env.base.queue, env.base.cfg.n_max).astype(np.int64)
    caps[~mask] = 0
    return observation.astype(np.float32), mask.astype(bool), caps


def residual_energy_metrics(energy, alive):
    """Describe the alive-node energy level and dispersion together."""
    values = np.asarray(energy, dtype=np.float64)[np.asarray(alive, dtype=bool)]
    if values.size == 0:
        return {
            "alive_nodes": 0,
            "residual_energy_fairness": 0.0,
            "residual_energy_cv": 0.0,
            "mean_residual_energy_j": 0.0,
            "min_residual_energy_j": 0.0,
            "p10_residual_energy_j": 0.0,
        }
    mean = float(values.mean())
    return {
        "alive_nodes": int(values.size),
        "residual_energy_fairness": FixedClusterTrainingEnv._jain(values),
        "residual_energy_cv": (
            float(values.std(ddof=0) / mean) if mean > 0.0 else 0.0
        ),
        "mean_residual_energy_j": mean,
        "min_residual_energy_j": float(values.min()),
        "p10_residual_energy_j": float(np.quantile(values, 0.10)),
    }


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
                "fnd_free_steps": int(
                    env.base.t_fnd if env.base.t_fnd is not None else steps
                ),
                "global_throughput": int(env.base.total_packets),
                "queue_fairness": FixedClusterTrainingEnv._jain(
                    env.cumulative_service
                ),
                "delivery_ratio": (
                    env.base.total_packets / env.base.total_packets_generated
                    if env.base.total_packets_generated > 0
                    else 0.0
                ),
                "target_ch_alive": bool(env.base.alive[int(info["target_ch"])]),
                **residual_energy_metrics(
                    env.base.energy, env.base.alive
                ),
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
        "mean_fnd_free_steps": float(
            np.mean([row["fnd_free_steps"] for row in rows])
        ),
        "mean_throughput": float(
            np.mean([row["global_throughput"] for row in rows])
        ),
        "mean_queue_fairness": float(
            np.mean([row["queue_fairness"] for row in rows])
        ),
        "mean_delivery_ratio": float(
            np.mean([row["delivery_ratio"] for row in rows])
        ),
        "mean_residual_energy_fairness": float(
            np.mean([row["residual_energy_fairness"] for row in rows])
        ),
        "mean_residual_energy_cv": float(
            np.mean([row["residual_energy_cv"] for row in rows])
        ),
        "mean_residual_energy_j": float(
            np.mean([row["mean_residual_energy_j"] for row in rows])
        ),
        "mean_min_residual_energy_j": float(
            np.mean([row["min_residual_energy_j"] for row in rows])
        ),
        "mean_p10_residual_energy_j": float(
            np.mean([row["p10_residual_energy_j"] for row in rows])
        ),
    }, first_env, first_observation


def reset_inspection_state(env):
    """Return an observation aligned with a freshly reset, active mask."""
    observation, mask, _ = env.reset()
    if not np.any(mask):
        raise RuntimeError(
            "inspection environment has no active member immediately after reset"
        )
    return observation

def policy_stability_summary(snapshots, relative_tolerance):
    metric_names = (
        "mean_fnd_free_steps",
        "mean_throughput",
        "mean_queue_fairness",
    )
    metrics = {}
    for name in metric_names:
        values = np.asarray(
            [row["evaluation"][name] for row in snapshots],
            dtype=np.float64,
        )
        mean = float(values.mean()) if values.size else 0.0
        relative_span = (
            float((values.max() - values.min()) / max(abs(mean), 1e-12))
            if values.size
            else None
        )
        metrics[name] = {
            "values": values.tolist(),
            "mean": mean,
            "relative_span": relative_span,
            "pass": bool(
                values.size >= 3 and relative_span <= relative_tolerance
            ),
        }
    assessed = len(snapshots) >= 3
    return {
        "assessed": assessed,
        "snapshot_count": len(snapshots),
        "relative_tolerance": float(relative_tolerance),
        "metrics": metrics,
        "pass": bool(assessed and all(row["pass"] for row in metrics.values())),
    }


def main():
    args = parse_args()
    if args.reinitialize_categorical_outputs and not args.initial_checkpoint:
        raise ValueError(
            "categorical output reinitialization requires an initial checkpoint"
        )
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
    reward_scale, reward_scale_evidence = load_reward_scale(
        args.reward_scale_config
    )
    qos_config, qos_evidence = load_qos_constraints(
        args.qos_constraint_config
    )
    qos_controller = QoSConstraintController(qos_config) if qos_config else None

    set_seeds(args.optimizer_seed)
    run_dir = ROOT / "outputs" / "phase2" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = run_dir / "episodes.jsonl"
    if episodes_path.exists():
        episodes_path.unlink()

    observation_schema = (
        PHASE2D_POLICY_SCHEMA
        if args.architecture == "equivariant_set_branching"
        else LEGACY_POLICY_SCHEMA
    )
    environments, curriculum_manifest, env_cfg = build_curriculum(
        seeds,
        args.max_steps,
        observation_schema=observation_schema,
        environment_profile=args.environment_profile,
    )
    max_branches = environments[0].base.n_nodes
    input_dims = {entry["input_dim"] for entry in curriculum_manifest}
    if len(input_dims) != 1:
        raise RuntimeError(f"inconsistent input dimensions: {input_dims}")
    input_dim = input_dims.pop()
    observation_layout = environments[0].observation_layout
    _, reward_payload = load_reward_model()
    reward_weights = dict(reward_payload["weights"])
    if args.idle_weight is not None:
        reward_weights["idle_energy_j"] = args.idle_weight
    if args.death_weight is not None:
        reward_weights["deaths"] = args.death_weight
    reward_payload = dict(reward_payload)
    reward_payload["weights"] = reward_weights
    reward_model = RewardModel(reward_payload["scales"], reward_weights)
    low_block, high_block, harvest_scale = trajectory_blocks(environments[0])
    agent_cfg = BranchingAgentConfig(
        input_dim=input_dim,
        actions=env_cfg.n_max + 1,
        budget=(
            env_cfg.frame_slot_budget
            if args.projection_budget is None
            else args.projection_budget
        ),
        replay_capacity=5000,
        max_branches=max_branches,
        architecture=args.architecture,
        state_schema=observation_schema,
        embedding_start_dim=int(observation_layout["embedding_start"]),
        learning_rate=args.learning_rate,
        normalize_input_blocks=args.normalize_input_blocks,
        hybrid_harvest_max_j=harvest_scale,
        trajectory_low_block=low_block,
        trajectory_high_block=high_block,
        trajectory_loss_weight=args.trajectory_loss_weight,
        concavity_loss_weight=args.concavity_loss_weight,
        trajectory_margin_fraction=args.trajectory_margin_fraction,
        demonstration_margin_loss_weight=args.demonstration_margin_loss_weight,
        demonstration_margin=args.demonstration_margin,
        precision=args.precision,
        reward_scale=reward_scale,
    )
    if reward_scale_evidence is not None:
        frozen_support = reward_scale_evidence["payload"]["support"]
        actual_support = {
            "v_min": agent_cfg.v_min,
            "v_max": agent_cfg.v_max,
            "atoms": agent_cfg.atoms,
        }
        if frozen_support != actual_support:
            raise ValueError(
                f"reward-scale support mismatch: {frozen_support} != "
                f"{actual_support}"
            )
    agent = BranchingDQNAgent(agent_cfg, device=args.device)
    initialization = None
    initialization_details = None
    if args.initial_checkpoint:
        checkpoint_path = Path(args.initial_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT / checkpoint_path
        checkpoint = torch.load(
            checkpoint_path, map_location=args.device, weights_only=False
        )
        checkpoint_architecture = checkpoint.get("config", {}).get(
            "architecture", "legacy_weight_tied"
        )
        if checkpoint_architecture != args.architecture:
            raise ValueError(
                "initial checkpoint architecture mismatch: "
                f"{checkpoint_architecture} != {args.architecture}"
            )
        checkpoint_config = checkpoint.get("config", {})
        checkpoint_schema = checkpoint_config.get(
            "state_schema", LEGACY_POLICY_SCHEMA
        )
        if checkpoint_schema != observation_schema:
            raise ValueError(
                "initial checkpoint state schema mismatch: "
                f"{checkpoint_schema} != {observation_schema}"
            )
        if int(checkpoint_config.get("input_dim", -1)) != input_dim:
            raise ValueError("initial checkpoint input dimension mismatch")
        if int(
            checkpoint_config.get("embedding_start_dim", 18)
        ) != int(observation_layout["embedding_start"]):
            raise ValueError(
                "initial checkpoint embedding boundary mismatch"
            )
        checkpoint_reward_scale = float(
            checkpoint_config.get("reward_scale", 1.0)
        )
        scale_changed = not np.isclose(
            checkpoint_reward_scale, reward_scale, rtol=0.0, atol=1e-15
        )
        if scale_changed and not args.reinitialize_categorical_outputs:
            raise ValueError(
                "checkpoint reward scale differs from this run; categorical "
                "output reinitialization is required"
            )
        agent.online.load_state_dict(checkpoint["online_state_dict"])
        agent.target.load_state_dict(checkpoint["target_state_dict"])
        if args.reinitialize_categorical_outputs:
            agent.reinitialize_categorical_outputs(seed=args.optimizer_seed)
        initialization = str(checkpoint_path.resolve())
        initialization_details = {
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_reward_scale": checkpoint_reward_scale,
            "run_reward_scale": reward_scale,
            "reward_scale_changed": bool(scale_changed),
            "categorical_outputs_reinitialized": bool(
                args.reinitialize_categorical_outputs
            ),
            "optimizer_reinitialized": True,
            "replay_reinitialized": True,
        }
        if qos_controller is not None:
            saved_qos = checkpoint.get("metadata", {}).get(
                "qos_constraints", {}
            ).get("controller_state")
            if saved_qos is not None:
                qos_controller.load_state_dict(saved_qos)
                initialization_details["qos_multipliers_restored"] = True

    expected_steps = args.episodes * args.max_steps
    rows = []
    stability_snapshots = []
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
        if qos_controller is not None:
            qos_controller.begin_episode()
        padded, padded_mask, caps = padded_state(
            env, observation, mask, max_branches
        )
        epsilon = epsilon_for(
            episode, args.episodes, args.epsilon_start, args.epsilon_end
        )
        episode_reward = 0.0
        episode_learning_reward = 0.0
        constrained_raw_reward = 0.0
        constraint_penalty_total = 0.0
        last_qos_details = None
        qos_episode_update = None
        raw_sums = {name: 0.0 for name in TERM_ORDER}
        weighted_sums = {name: 0.0 for name in TERM_ORDER}
        losses = []
        loss_terms = []
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
            constraint_penalty = 0.0
            if qos_controller is not None:
                constraint_penalty, last_qos_details = qos_controller.evaluate_info(info)
            constrained_step_reward = reward + constraint_penalty
            scaled_reward = learning_reward(
                constrained_step_reward, reward_scale
            )
            if (
                not np.isfinite(reward)
                or not np.isfinite(constraint_penalty)
                or not np.isfinite(scaled_reward)
                or not np.all(np.isfinite(next_padded))
            ):
                stopped_for_nonfinite = True
                done = True
                break
            agent.store(
                padded,
                action,
                scaled_reward,
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
                loss_terms.append(dict(agent.last_loss_terms))
            episode_reward += reward
            constrained_raw_reward += constrained_step_reward
            constraint_penalty_total += constraint_penalty
            episode_learning_reward += scaled_reward
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

        if qos_controller is not None:
            qos_episode_update = qos_controller.end_episode()
            if last_qos_details is not None:
                last_qos_details["episode_end_update"] = qos_episode_update

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
            "raw_physical_reward": episode_reward,
            "constrained_raw_learning_reward": constrained_raw_reward,
            "constraint_penalty_total": constraint_penalty_total,
            "qos_constraint": last_qos_details,
            "qos_episode_update": qos_episode_update,
            "scaled_learning_reward": episode_learning_reward,
            "reward_scale": reward_scale,
            "target_packets": packets,
            "packets_per_step": packets / max(1, steps),
            "zero_action_fraction": zero_action_steps / max(1, steps),
            "mean_allocated_slots": allocated_slots / max(1, steps),
            "mean_loss": float(np.mean(losses)) if losses else None,
            "mean_loss_terms": (
                {
                    name: float(np.mean([item[name] for item in loss_terms]))
                    for name in loss_terms[0]
                }
                if loss_terms else None
            ),
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

        completed = episode + 1
        stability_start = max(
            args.stability_interval,
            args.episodes - args.stability_tail_episodes,
        )
        if completed >= stability_start and (
            completed % args.stability_interval == 0
            or completed == args.episodes
        ):
            snapshot_evaluation, _, _ = greedy_curriculum_evaluation(
                agent, environments, max_branches, reward_model
            )
            snapshot = {
                "episode": completed,
                "evaluation": snapshot_evaluation,
            }
            stability_snapshots.append(snapshot)
            agent.save(
                run_dir / f"stability_episode_{completed}.pt",
                snapshot,
            )
            print(
                f"POLICY_STABILITY_SNAPSHOT={completed} "
                f"FND_FREE={snapshot_evaluation['mean_fnd_free_steps']:.4f} "
                f"THROUGHPUT={snapshot_evaluation['mean_throughput']:.4f} "
                f"FAIRNESS={snapshot_evaluation['mean_queue_fairness']:.6f}",
                flush=True,
            )

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
    collapse = bool(
        stopped_for_collapse
        or evaluation["mean_zero_action_fraction"] > 0.80
        or evaluation["mean_packets"] <= 0.0
    )

    convergence = {"assessed": len(rows) >= 100, "relative_change": None, "pass": False}
    if len(rows) >= 100:
        convergence_field = (
            "constrained_raw_learning_reward" if qos_controller is not None
            else "reward"
        )
        previous = np.mean([row[convergence_field] for row in rows[-100:-50]])
        current = np.mean([row[convergence_field] for row in rows[-50:]])
        relative = abs(current - previous) / max(1.0, abs(previous))
        convergence = {
            "assessed": True,
            "field": convergence_field,
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
        and policy_stability["pass"]
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
        "initialization_details": initialization_details,
        "reward_scale": reward_scale,
        "reward_scale_evidence": reward_scale_evidence,
        "qos_constraints": {
            "enabled": qos_controller is not None,
            "evidence": qos_evidence,
            "controller_state": (
                qos_controller.state_dict()
                if qos_controller is not None else None
            ),
        },
        "reward_application": {
            "replay_and_c51_reward_scaled": True,
            "physical_metrics_scaled": False,
            "episode_reward_field": "raw_physical_reward",
            "constraint_application": "raw_learning_reward_before_frozen_c51_scale",
            "learning_reward_field": "scaled_learning_reward",
        },
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "projection_budget": agent_cfg.budget,
        "environment_capacity": env_cfg.frame_slot_budget,
        "development_seeds": seeds,
        "schedule_schema_version": curriculum_manifest[0]["schedule_schema_version"],
        "held_out_pilot_seeds": sorted(held_out),
        "held_out_overlap": sorted(overlap),
        "git_hash": git_hash(),
        "environment_profile": curriculum_manifest[0].get(
            "environment_profile"
        ),
        "curriculum_clusters": curriculum_manifest,
        "curriculum_pair_count": len(environments),
        "full_curriculum_seen": full_curriculum_seen,
        "max_padded_branches": max_branches,
        "agent_config": agent_cfg.__dict__,
        "online_parameter_count": int(
            sum(parameter.numel() for parameter in agent.online.parameters())
        ),
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
        "policy_stability": policy_stability,
        "policy_stability_snapshots": stability_snapshots,
        "scope": (
            "Development curriculum uses an exogenous per-round CH schedule; "
            "held-out Phase 3 seeds are excluded and CH selection remains "
            "outside the learned intervention. Paper-aligned profiles are "
            "exploratory and do not replace the registered frozen HEART-CH "
            "experiment."
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
    print(f"POLICY_STABILITY_PASS={policy_stability['pass']}")
    print(f"PHASE2_CURRICULUM_GATE_PASS={gate_pass}")
    print(f"summary={summary_path}")
    return 0 if status in {"smoke_pass", "pass"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
