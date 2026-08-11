"""Development-only Phase 2C branch-identity robustness gate.

The audit moves the complete state/mask/cap/action-mask bundle, performs
inference in the permuted layout, and inverse-maps allocations before the
physical environment is stepped.  It never uses held-out seeds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.branch_permutation import (
    action_mask_from_caps,
    active_branch_permutation,
    inverse_map_branch_values,
    permute_complete_bundle,
    swap_permutation,
)
from agents.branching_dqn import BranchingAgentConfig, BranchingDQNAgent
from envs.fixed_cluster_training_env import (
    FixedClusterTrainingEnv,
    SOLAR_DECLINING_STATES,
    SOLAR_HIGH_HARVEST_STATES,
)
from experiments.audit_phase2_mid_episode_hybrid_sensitivity import hybrid_pair
from experiments.train_phase2_dynamic_curriculum import (
    build_curriculum,
    padded_state,
    residual_energy_metrics,
)
from experiments.train_phase2_fixed_cluster import load_reward_model


CHECKPOINTS = {
    2299: (
        "HTA_MAC_Phase2B_Confirmation_Results_20260803/runs/"
        "phase2b_confirm_shared_b12_seed2299_125ep/branching_c51.pt",
        "F67962F4F48871D7A7BA9446F1E528A6AE381305CED4E9207DB68551392C8049",
    ),
    3299: (
        "HTA_MAC_Phase2B_Confirmation_Results_20260803/runs/"
        "phase2b_confirm_shared_b12_seed3299_125ep/branching_c51.pt",
        "B1CFBC377BE1A5BAE6FE0D571CD3AF7DE06522A0C7DA45CB4A8888376632F182",
    ),
    4299: (
        "HTA_MAC_Phase2B_Confirmation_Results_20260803/runs/"
        "phase2b_confirm_shared_b12_seed4299_125ep/branching_c51.pt",
        "350B443B3CE84F47A2DC6E6377D86C08010C8375091E6610E33019E0A5C6DE61",
    ),
}

GATES = {
    "random_inverse_projected_allocation_agreement_min": 0.95,
    "targeted_inverse_projected_allocation_agreement_min": 0.90,
    "state_conditioned_classification_fraction_min": 0.50,
    "fixed_identity_classification_fraction_max": 0.10,
    "require_hybrid_high_more_than_high_fewer": True,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "phase2" / "phase2c_branch_identity_audit_20260804",
    )
    parser.add_argument("--development-seeds", default="2300,2301,2302,2303,2304")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--random-permutations", type=int, default=20)
    parser.add_argument("--targeted-swaps", type=int, default=10)
    parser.add_argument("--hybrid-probes", type=int, default=20)
    parser.add_argument("--energy-tolerance-j", type=float, default=0.05)
    parser.add_argument("--audit-seed", type=int, default=20260804)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_agent(path: Path) -> BranchingDQNAgent:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config_data = dict(payload["config"])
    config_data["precision"] = "fp32"
    agent = BranchingDQNAgent(BranchingAgentConfig(**config_data), device="cpu")
    agent.online.load_state_dict(payload["online_state_dict"])
    agent.online.eval()
    return agent


def inference(
    agent, state, mask, caps, tie_break_priorities=None
):
    state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
    mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
    with torch.no_grad():
        q_values = agent.q_values_tensor(state_t, mask_t)[0].cpu().numpy()
    action = agent._project(
        q_values,
        mask,
        caps=caps,
        tie_break_priorities=tie_break_priorities,
    )
    local = np.argmax(q_values, axis=1)
    return action, q_values, local


def permuted_inference(agent, state, mask, caps, permutation):
    action_mask = action_mask_from_caps(mask, caps, agent.cfg.actions)
    priorities = np.arange(len(state), dtype=np.int64)
    bundle = permute_complete_bundle(
        state,
        mask,
        caps,
        action_mask,
        permutation,
        tie_break_priorities=priorities,
    )
    branch_action, branch_q, branch_local = inference(
        agent,
        bundle["state"],
        bundle["mask"],
        bundle["caps"],
        bundle["tie_break_priorities"],
    )
    return {
        "branch_action": branch_action,
        "branch_q": branch_q,
        "branch_local": branch_local,
        "physical_action": inverse_map_branch_values(branch_action, permutation),
        "physical_q": inverse_map_branch_values(branch_q, permutation),
        "physical_local": inverse_map_branch_values(branch_local, permutation),
    }


def safe_fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def classify_identity(canonical, branch, physical):
    raw_distance = float(np.abs(branch - canonical).sum())
    inverse_distance = float(np.abs(physical - canonical).sum())
    if inverse_distance < raw_distance:
        label = "state_conditioned"
    elif raw_distance < inverse_distance:
        label = "fixed_branch_identity"
    else:
        label = "tie"
    return label, raw_distance, inverse_distance


def compare_permutation(agent, probe, kind, index):
    canonical_action, canonical_q, canonical_local = inference(
        agent, probe["state"], probe["mask"], probe["caps"]
    )
    moved = permuted_inference(
        agent,
        probe["state"],
        probe["mask"],
        probe["caps"],
        probe["permutation"],
    )
    active = probe["mask"]
    label, raw_distance, inverse_distance = classify_identity(
        canonical_action, moved["branch_action"], moved["physical_action"]
    )
    canonical_marginal = np.diff(canonical_q, axis=1)
    raw_marginal = np.diff(moved["branch_q"], axis=1)
    physical_marginal = np.diff(moved["physical_q"], axis=1)
    record = {
        "kind": kind,
        "probe_index": int(index),
        "seed": int(probe["seed"]),
        "target_rank": int(probe["target_rank"]),
        "round": int(probe["round"]),
        "active_branches": int(active.sum()),
        "raw_local_argmax_agreement": float(
            np.mean(moved["branch_local"][active] == canonical_local[active])
        ),
        "inverse_local_argmax_agreement": float(
            np.mean(moved["physical_local"][active] == canonical_local[active])
        ),
        "raw_projected_action_agreement": float(
            np.mean(moved["branch_action"][active] == canonical_action[active])
        ),
        "inverse_projected_allocation_agreement": float(
            np.mean(moved["physical_action"][active] == canonical_action[active])
        ),
        "inverse_projected_allocation_disagreement": float(
            np.mean(moved["physical_action"][active] != canonical_action[active])
        ),
        "raw_allocation_l1": raw_distance,
        "inverse_allocation_l1": inverse_distance,
        "identity_classification": label,
        "raw_marginal_q_mae": float(
            np.mean(np.abs(raw_marginal[active] - canonical_marginal[active]))
        ),
        "inverse_marginal_q_mae": float(
            np.mean(np.abs(physical_marginal[active] - canonical_marginal[active]))
        ),
    }
    if kind == "targeted_swap":
        high, declining = probe["pair"]
        record.update(
            {
                "high_node": int(high),
                "declining_node": int(declining),
                "equal_cap": bool(probe["caps"][high] == probe["caps"][declining]),
                "energy_difference_j": float(probe["energy_difference_j"]),
                "canonical_high_slots": int(canonical_action[high]),
                "canonical_declining_slots": int(canonical_action[declining]),
                "inverse_high_slots": int(moved["physical_action"][high]),
                "inverse_declining_slots": int(moved["physical_action"][declining]),
            }
        )
    return record


def find_target_pair(env, mask, caps, tolerance):
    solar = env.base.solar_states
    eligible = mask & (caps > 0)
    highs = np.flatnonzero(eligible & np.isin(solar, SOLAR_HIGH_HARVEST_STATES))
    lows = np.flatnonzero(eligible & np.isin(solar, SOLAR_DECLINING_STATES))
    candidates = []
    for high in highs:
        for low in lows:
            if caps[high] != caps[low]:
                continue
            difference = abs(float(env.base.energy[high] - env.base.energy[low]))
            if difference <= tolerance:
                candidates.append((difference, int(high), int(low)))
    return min(candidates) if candidates else None


def collect_probes(agent, environments, args, seed_offset):
    rng = np.random.default_rng(args.audit_seed + seed_offset)
    random_probes, targeted_probes, hybrid_probes = [], [], []
    for env in environments:
        observation, mask, _ = env.reset()
        done = False
        round_index = 0
        while not done:
            state, active, caps = padded_state(env, observation, mask, env.base.n_nodes)
            if active.sum() > 1 and len(random_probes) < args.random_permutations:
                permutation = active_branch_permutation(active, rng)
                if not np.array_equal(permutation, np.arange(permutation.size)):
                    random_probes.append(
                        {
                            "state": state.copy(), "mask": active.copy(),
                            "caps": caps.copy(), "permutation": permutation,
                            "seed": env.seed, "target_rank": env.target_rank,
                            "round": round_index,
                        }
                    )
            pair = find_target_pair(env, active, caps, args.energy_tolerance_j)
            if pair is not None and len(targeted_probes) < args.targeted_swaps:
                difference, high, declining = pair
                targeted_probes.append(
                    {
                        "state": state.copy(), "mask": active.copy(),
                        "caps": caps.copy(),
                        "permutation": swap_permutation(len(active), high, declining),
                        "pair": (high, declining),
                        "energy_difference_j": difference,
                        "seed": env.seed, "target_rank": env.target_rank,
                        "round": round_index,
                    }
                )
            eligible = np.flatnonzero(active & (caps >= 1))
            if eligible.size and len(hybrid_probes) < args.hybrid_probes:
                node = int(eligible[len(hybrid_probes) % eligible.size])
                low, high_state = hybrid_pair(env, state, node)
                hybrid_probes.append(
                    {"low": low, "high": high_state, "mask": active.copy(),
                     "caps": caps.copy(), "node": node}
                )
            action, _, _ = inference(agent, state, active, caps)
            observation, mask, done, _ = env.step(action)
            round_index += 1
            if (
                len(random_probes) >= args.random_permutations
                and len(targeted_probes) >= args.targeted_swaps
                and len(hybrid_probes) >= args.hybrid_probes
            ):
                return random_probes, targeted_probes, hybrid_probes
    return random_probes, targeted_probes, hybrid_probes


def hybrid_summary(agent, probes):
    high_more = high_fewer = equal = ordered = 0
    for probe in probes:
        low_action, low_q, _ = inference(agent, probe["low"], probe["mask"], probe["caps"])
        high_action, high_q, _ = inference(agent, probe["high"], probe["mask"], probe["caps"])
        node = probe["node"]
        high_more += int(high_action[node] > low_action[node])
        high_fewer += int(high_action[node] < low_action[node])
        equal += int(high_action[node] == low_action[node])
        ordered += int(np.all(np.diff(high_q[node]) >= np.diff(low_q[node])))
    return {
        "probe_count": len(probes),
        "high_more": high_more,
        "high_fewer": high_fewer,
        "equal": equal,
        "all_marginals_ordered": ordered,
    }


def summarize_records(records):
    count = len(records)
    labels = {name: sum(row["identity_classification"] == name for row in records)
              for name in ("state_conditioned", "fixed_branch_identity", "tie")}
    metrics = (
        "raw_local_argmax_agreement", "inverse_local_argmax_agreement",
        "raw_projected_action_agreement", "inverse_projected_allocation_agreement",
        "inverse_projected_allocation_disagreement", "raw_allocation_l1",
        "inverse_allocation_l1", "raw_marginal_q_mae", "inverse_marginal_q_mae",
    )
    result = {"probe_count": count, "identity_classifications": labels}
    result["identity_classification_fractions"] = {
        name: safe_fraction(value, count) for name, value in labels.items()
    }
    for metric in metrics:
        values = [float(row[metric]) for row in records]
        result[f"mean_{metric}"] = float(np.mean(values)) if values else math.nan
    return result


def rollout_metrics(agent, environments, reward_model, permuted, audit_seed):
    rows = []
    for environment_index, env in enumerate(environments):
        rng = np.random.default_rng(audit_seed + environment_index)
        observation, mask, _ = env.reset()
        done = False
        reward_total = packets = allocated = steps = 0
        while not done:
            state, active, caps = padded_state(env, observation, mask, env.base.n_nodes)
            if permuted and active.sum() > 1:
                order = active_branch_permutation(active, rng)
                result = permuted_inference(agent, state, active, caps, order)
                action = result["physical_action"]
            else:
                action, _, _ = inference(agent, state, active, caps)
            observation, mask, done, info = env.step(action)
            reward, _ = reward_model.evaluate(info["reward_raw_terms"])
            reward_total += reward
            packets += int(info["target_packets_delivered"])
            allocated += int(action.sum())
            steps += 1
        energy = residual_energy_metrics(env.base.energy, env.base.alive)
        rows.append(
            {
                "seed": env.seed, "target_rank": env.target_rank,
                "reward": reward_total, "packets": packets, "steps": steps,
                "mean_allocated_slots": safe_fraction(allocated, steps),
                "delivery_ratio": safe_fraction(
                    env.base.total_packets, env.base.total_packets_generated
                ),
                "fnd_observed": env.base.t_fnd is not None,
                "fnd_round": env.base.t_fnd,
                "packet_jain_fairness": FixedClusterTrainingEnv._jain(
                    env.cumulative_service
                ),
                **energy,
            }
        )
    mean_fields = (
        "reward", "packets", "mean_allocated_slots", "delivery_ratio",
        "packet_jain_fairness", "residual_energy_fairness", "residual_energy_cv",
        "mean_residual_energy_j", "min_residual_energy_j", "p10_residual_energy_j",
    )
    return {
        "mode": "complete_bundle_permuted_inverse_mapped" if permuted else "canonical",
        "environment_count": len(rows),
        "fnd_event_count": sum(row["fnd_observed"] for row in rows),
        "fnd_right_censored_count": sum(not row["fnd_observed"] for row in rows),
        **{f"mean_{field}": float(np.mean([row[field] for row in rows]))
           for field in mean_fields},
        "rows": rows,
    }


def checkpoint_gate(random_summary, targeted_summary, hybrid):
    checks = {
        "random_inverse_allocation_agreement": (
            random_summary["mean_inverse_projected_allocation_agreement"]
            >= GATES["random_inverse_projected_allocation_agreement_min"]
        ),
        "targeted_inverse_allocation_agreement": (
            targeted_summary["mean_inverse_projected_allocation_agreement"]
            >= GATES["targeted_inverse_projected_allocation_agreement_min"]
        ),
        "state_conditioned_classification": (
            targeted_summary["identity_classification_fractions"]["state_conditioned"]
            >= GATES["state_conditioned_classification_fraction_min"]
        ),
        "fixed_identity_classification": (
            targeted_summary["identity_classification_fractions"]["fixed_branch_identity"]
            <= GATES["fixed_identity_classification_fraction_max"]
        ),
        "hybrid_direction": hybrid["high_more"] > hybrid["high_fewer"],
    }
    return checks, all(checks.values())


def write_csv(path, records):
    if not records:
        return
    fields = sorted({key for row in records for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main():
    args = parse_args()
    if args.random_permutations < 20 or args.targeted_swaps < 10:
        raise ValueError("predeclared minimum is 20 random permutations and 10 targeted swaps")
    seeds = [int(value) for value in args.development_seeds.split(",")]
    if set(seeds) != {2300, 2301, 2302, 2303, 2304}:
        raise ValueError("identity gate is frozen to development seeds 2300-2304")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reward_model, reward_payload = load_reward_model()
    runs = []
    all_records = []
    for optimizer_seed, (relative, expected_hash) in CHECKPOINTS.items():
        checkpoint = ROOT / relative
        observed_hash = sha256_file(checkpoint)
        if observed_hash != expected_hash:
            raise RuntimeError(f"checkpoint hash mismatch for seed {optimizer_seed}")
        agent = load_agent(checkpoint)
        environments, manifest, _ = build_curriculum(
            seeds, args.max_steps,
            observation_schema=agent.cfg.state_schema,
        )
        random_probes, targeted_probes, hybrid_probes = collect_probes(
            agent, environments, args, optimizer_seed
        )
        if len(random_probes) < args.random_permutations:
            raise RuntimeError(f"insufficient random probes for seed {optimizer_seed}")
        if len(targeted_probes) < args.targeted_swaps:
            raise RuntimeError(f"insufficient targeted swaps for seed {optimizer_seed}")
        random_records = [
            {"optimizer_seed": optimizer_seed, **compare_permutation(agent, probe, "random", i)}
            for i, probe in enumerate(random_probes)
        ]
        targeted_records = [
            {"optimizer_seed": optimizer_seed, **compare_permutation(agent, probe, "targeted_swap", i)}
            for i, probe in enumerate(targeted_probes)
        ]
        all_records.extend(random_records + targeted_records)
        random_summary = summarize_records(random_records)
        targeted_summary = summarize_records(targeted_records)
        hybrid = hybrid_summary(agent, hybrid_probes)
        canonical_envs, _, _ = build_curriculum(
            seeds, args.max_steps,
            observation_schema=agent.cfg.state_schema,
        )
        permuted_envs, _, _ = build_curriculum(
            seeds, args.max_steps,
            observation_schema=agent.cfg.state_schema,
        )
        canonical = rollout_metrics(
            agent, canonical_envs, reward_model, False, args.audit_seed + optimizer_seed
        )
        permuted = rollout_metrics(
            agent, permuted_envs, reward_model, True, args.audit_seed + optimizer_seed
        )
        checks, passed = checkpoint_gate(random_summary, targeted_summary, hybrid)
        runs.append(
            {
                "optimizer_seed": optimizer_seed,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": observed_hash,
                "random_permutations": random_summary,
                "targeted_swaps": targeted_summary,
                "hybrid_response": hybrid,
                "canonical_rollout": canonical,
                "permuted_inverse_mapped_rollout": permuted,
                "gate_checks": checks,
                "gate_pass": passed,
                "curriculum_manifest": manifest,
            }
        )
    gate_pass = all(run["gate_pass"] for run in runs)
    report = {
        "status": "gate_pass" if gate_pass else "gate_fail",
        "scope": "development-only; no held-out seeds; frozen HEART-CH schedule replay",
        "held_out_seeds_used": False,
        "development_seeds": seeds,
        "audit_seed": args.audit_seed,
        "random_permutations_per_checkpoint": args.random_permutations,
        "targeted_swaps_per_checkpoint": args.targeted_swaps,
        "target_energy_tolerance_j": args.energy_tolerance_j,
        "complete_bundle": ["state_row", "validity_mask", "queue_cap", "action_mask"],
        "physical_action_inverse_mapped_before_step": True,
        "predeclared_gates": GATES,
        "reward_configuration": reward_payload,
        "runs": runs,
    }
    records_path = args.output_dir / "branch_identity_records.csv"
    report_path = args.output_dir / "branch_identity_audit.json"
    write_csv(records_path, all_records)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    hashes = {
        records_path.name: sha256_file(records_path),
        report_path.name: sha256_file(report_path),
    }
    hashes_path = args.output_dir / "artifact_hashes.json"
    hashes_path.write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print(f"STATUS={report['status']}")
    for run in runs:
        random = run["random_permutations"]
        targeted = run["targeted_swaps"]
        hybrid = run["hybrid_response"]
        print(
            f"SEED={run['optimizer_seed']} "
            f"RANDOM_INVERSE_AGREEMENT={random['mean_inverse_projected_allocation_agreement']:.6f} "
            f"TARGET_INVERSE_AGREEMENT={targeted['mean_inverse_projected_allocation_agreement']:.6f} "
            f"STATE_CONDITIONED={targeted['identity_classification_fractions']['state_conditioned']:.6f} "
            f"FIXED_IDENTITY={targeted['identity_classification_fractions']['fixed_branch_identity']:.6f} "
            f"HIGH_MORE={hybrid['high_more']} HIGH_FEWER={hybrid['high_fewer']} "
            f"PASS={run['gate_pass']}"
        )
    print(f"REPORT={report_path}")
    print(f"HASHES={hashes_path}")
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
