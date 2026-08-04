"""Calibrate Phase 2C C51 reward scale from fixed development rollouts.

This script is development-only. It validates three repaired checkpoints, runs
all 25 schedule/target-rank curriculum pairs for each checkpoint, archives every
per-step reward and discounted return, and derives one conservative reward
scale. Held-out seeds are forbidden.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.reward_model import TERM_ORDER, RewardModel
from experiments.audit_phase2_trajectory_sensitivity import load_agent
from experiments.train_phase2_dynamic_curriculum import build_curriculum, padded_state
from experiments.train_phase2_fixed_cluster import load_reward_model


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    result = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        result[index] = running
    return result


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile of an empty sample")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must be in [0, 1]")
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def describe(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size or not np.all(np.isfinite(array)):
        raise ValueError("return sample is empty or non-finite")
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "q0_005": float(np.quantile(array, 0.005)),
        "q0_01": float(np.quantile(array, 0.01)),
        "q0_50": float(np.quantile(array, 0.50)),
        "q0_95": float(np.quantile(array, 0.95)),
        "q0_99": float(np.quantile(array, 0.99)),
        "q0_995": float(np.quantile(array, 0.995)),
        "maximum": float(array.max()),
    }


def choose_scale(q_star: float, v_max: float, headroom: float) -> float:
    if not math.isfinite(q_star) or q_star <= 0.0:
        raise ValueError("q_star must be finite and positive")
    if v_max <= 0.0 or not 0.0 < headroom < 1.0:
        raise ValueError("invalid support/headroom configuration")
    return float(min(1.0, headroom * v_max / q_star))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "config" / "phase2c_reference_rollouts.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "phase2" / "phase2c_return_support",
    )
    return parser.parse_args()


def resolved(root: Path, path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else root / path


def validate_manifest(manifest: dict) -> None:
    seeds = {int(value) for value in manifest["development_seeds"]}
    forbidden = {int(value) for value in manifest["held_out_seeds_forbidden"]}
    overlap = seeds & forbidden
    if overlap:
        raise ValueError(f"held-out seeds entered calibration: {sorted(overlap)}")
    if seeds != {2300, 2301, 2302, 2303, 2304}:
        raise ValueError(f"unexpected development seed set: {sorted(seeds)}")
    if len(manifest["checkpoints"]) != 3:
        raise ValueError("exactly three repaired checkpoints are required")
    if int(manifest["max_steps"]) != 300:
        raise ValueError("registered reference horizon must remain 300")
    if float(manifest["rollout_epsilon"]) != 0.10:
        raise ValueError("registered reference epsilon must remain 0.10")


def checkpoint_payload_and_agent(entry: dict, manifest: dict):
    path = resolved(ROOT, entry["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_hash = sha256_file(path)
    expected_hash = str(entry["sha256"]).upper()
    if observed_hash != expected_hash:
        raise RuntimeError(
            f"checkpoint hash mismatch for {path}: {observed_hash} != {expected_hash}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]
    expected_support = manifest["expected_support"]
    checks = {
        "v_min": float(expected_support["v_min"]),
        "v_max": float(expected_support["v_max"]),
        "atoms": int(expected_support["atoms"]),
        "budget": int(manifest["projection_budget"]),
    }
    for name, expected in checks.items():
        observed = config[name]
        if observed != expected:
            raise RuntimeError(
                f"checkpoint {entry['optimizer_seed']} {name} mismatch: "
                f"{observed} != {expected}"
            )
    metadata_reward = payload.get("metadata", {}).get("reward_calibration")
    _, reward_payload = load_reward_model()
    if metadata_reward != reward_payload:
        raise RuntimeError(
            f"checkpoint {entry['optimizer_seed']} reward calibration mismatch"
        )
    return path, observed_hash, payload, load_agent(path)


def rollout_one(
    agent,
    environment,
    reward_model: RewardModel,
    epsilon: float,
    rollout_seed: int,
) -> list[dict]:
    np.random.seed(rollout_seed)
    torch.manual_seed(rollout_seed)
    observation, mask, _ = environment.reset()
    records = []
    step = 0
    done = False
    while not done:
        state, active_mask, caps = padded_state(
            environment, observation, mask, environment.base.n_nodes
        )
        action, _ = agent.act(
            state, active_mask, epsilon=epsilon, caps=caps
        )
        next_observation, next_mask, done, info = environment.step(action)
        reward, weighted = reward_model.evaluate(info["reward_raw_terms"])
        if not math.isfinite(reward):
            raise RuntimeError("non-finite rollout reward")
        record = {
            "step": step,
            "reward": float(reward),
            "done": int(done),
            "allocated_slots": int(np.asarray(action).sum()),
            "target_packets_delivered": int(info["target_packets_delivered"]),
            "target_cluster": int(info["target_cluster"]),
            "target_ch": int(info["target_ch"]),
            "target_member_count": int(len(info["target_members"])),
        }
        for name in TERM_ORDER:
            record[f"raw_{name}"] = float(info["reward_raw_terms"][name])
            record[f"weighted_{name}"] = float(weighted[name])
        records.append(record)
        observation, mask = next_observation, next_mask
        step += 1
        if step > environment.base.cfg.max_rounds:
            raise RuntimeError("rollout exceeded registered environment horizon")
    return records


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    manifest_hash = sha256_file(manifest_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    development_seeds = [int(value) for value in manifest["development_seeds"]]
    environments, curriculum_manifest, _ = build_curriculum(
        development_seeds, int(manifest["max_steps"])
    )
    if len(environments) != 25:
        raise RuntimeError(f"expected 25 curriculum pairs, found {len(environments)}")
    _, reward_payload = load_reward_model()
    reward_model = RewardModel(reward_payload["scales"], reward_payload["weights"])

    all_rows: list[dict] = []
    grouped_returns: dict[tuple[int, int], list[float]] = defaultdict(list)
    checkpoint_records = []
    for checkpoint_index, entry in enumerate(manifest["checkpoints"]):
        checkpoint_path, checkpoint_hash, payload, agent = checkpoint_payload_and_agent(
            entry, manifest
        )
        optimizer_seed = int(entry["optimizer_seed"])
        gamma = float(payload["config"]["gamma"])
        checkpoint_records.append(
            {
                "optimizer_seed": optimizer_seed,
                "path": str(checkpoint_path),
                "sha256": checkpoint_hash,
                "gamma": gamma,
            }
        )
        for environment_index, environment in enumerate(environments):
            rollout_seed = (
                int(manifest["audit_seed"])
                + checkpoint_index * 100_000
                + int(environment.seed) * 10
                + int(environment.target_rank)
            )
            trajectory = rollout_one(
                agent,
                environment,
                reward_model,
                float(manifest["rollout_epsilon"]),
                rollout_seed,
            )
            returns = discounted_returns(
                [float(row["reward"]) for row in trajectory], gamma
            )
            for row, value in zip(trajectory, returns):
                row.update(
                    {
                        "optimizer_seed": optimizer_seed,
                        "checkpoint_sha256": checkpoint_hash,
                        "schedule_seed": int(environment.seed),
                        "target_rank": int(environment.target_rank),
                        "rollout_seed": rollout_seed,
                        "gamma": gamma,
                        "discounted_return": float(value),
                    }
                )
                all_rows.append(row)
            grouped_returns[(optimizer_seed, int(environment.seed))].extend(returns)

    if not all_rows:
        raise RuntimeError("reference rollout produced no transitions")
    csv_path = output_dir / "phase2c_reference_returns.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    csv_hash = sha256_file(csv_path)

    group_stats = []
    probability = float(manifest["quantile_probability"])
    for (optimizer_seed, schedule_seed), values in sorted(grouped_returns.items()):
        stats = describe(values)
        stats.update(
            {
                "optimizer_seed": optimizer_seed,
                "schedule_seed": schedule_seed,
                "selected_upper_quantile": quantile(values, probability),
            }
        )
        group_stats.append(stats)
    q_star_record = max(
        group_stats, key=lambda item: float(item["selected_upper_quantile"])
    )
    q_star = float(q_star_record["selected_upper_quantile"])
    support = manifest["expected_support"]
    v_min = float(support["v_min"])
    v_max = float(support["v_max"])
    headroom = float(manifest["headroom_fraction"])
    reward_scale = choose_scale(q_star, v_max, headroom)
    raw_returns = [float(row["discounted_return"]) for row in all_rows]
    scaled_returns = [reward_scale * value for value in raw_returns]
    raw_stats = describe(raw_returns)
    scaled_stats = describe(scaled_returns)
    outside = [value for value in scaled_returns if value < v_min or value > v_max]
    lower_headroom = headroom * v_min
    upper_headroom = headroom * v_max
    gates = {
        "all_values_finite": all(math.isfinite(value) for value in raw_returns),
        "upper_q0_995_within_headroom": scaled_stats["q0_995"] <= upper_headroom + 1e-9,
        "lower_q0_005_within_headroom": scaled_stats["q0_005"] >= lower_headroom - 1e-9,
        "zero_scaled_returns_outside_support": len(outside) == 0,
        "checkpoint_count_is_three": len(checkpoint_records) == 3,
        "curriculum_pair_count_is_25": len(curriculum_manifest) == 25,
    }
    gate_pass = all(gates.values())
    report = {
        "status": "gate_pass" if gate_pass else "gate_fail_do_not_train",
        "scope": "development-only; held-out seeds forbidden",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "returns_csv": str(csv_path),
        "returns_csv_sha256": csv_hash,
        "checkpoints": checkpoint_records,
        "development_seeds": development_seeds,
        "curriculum_pair_count": len(curriculum_manifest),
        "transition_count": len(all_rows),
        "quantile_probability": probability,
        "q_star": q_star,
        "q_star_optimizer_seed": int(q_star_record["optimizer_seed"]),
        "q_star_schedule_seed": int(q_star_record["schedule_seed"]),
        "headroom_fraction": headroom,
        "reward_scale": reward_scale,
        "support": support,
        "raw_return_distribution": raw_stats,
        "scaled_return_distribution": scaled_stats,
        "scaled_returns_outside_support_count": len(outside),
        "scaled_returns_outside_support_fraction": len(outside) / len(scaled_returns),
        "per_checkpoint_schedule_seed": group_stats,
        "gates": gates,
    }
    report_path = output_dir / "phase2c_return_support_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if gate_pass:
        frozen = {
            "schema_version": 1,
            "status": "frozen_development_scale",
            "reward_scale": reward_scale,
            "apply_to": "replay_and_c51_reward_only",
            "physical_metrics_scaled": False,
            "q_star": q_star,
            "q_star_optimizer_seed": int(q_star_record["optimizer_seed"]),
            "q_star_schedule_seed": int(q_star_record["schedule_seed"]),
            "manifest_sha256": manifest_hash,
            "returns_csv_sha256": csv_hash,
            "checkpoint_sha256": [item["sha256"] for item in checkpoint_records],
            "support": support,
            "headroom_fraction": headroom,
            "quantile_probability": probability,
            "held_out_seeds_used": False,
        }
        frozen_path = ROOT / "config" / "phase2c_return_scale.json"
        frozen_path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
        print(f"PHASE2C_RETURN_SCALE_GATE_PASS c={reward_scale:.12g}")
        print(f"FROZEN {frozen_path}")
    else:
        print("PHASE2C_RETURN_SCALE_GATE_FAIL")
        print(json.dumps(gates, indent=2))
    print(f"REPORT {report_path}")
    print(f"RETURNS {csv_path}")
    if not gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()