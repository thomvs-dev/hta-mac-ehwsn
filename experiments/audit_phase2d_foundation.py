"""Audit trained Phase 2D symmetry, feasibility, and C51 boundary occupancy."""

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

from agents.branch_permutation import (
    action_mask_from_caps,
    active_branch_permutation,
    inverse_map_branch_values,
    permute_complete_bundle,
    swap_permutation,
)
from agents.branching_dqn import BranchingAgentConfig, BranchingDQNAgent
from envs.policy_observation import PHASE2D_POLICY_SCHEMA
from experiments.train_phase2_dynamic_curriculum import (
    build_curriculum,
    padded_state,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument(
        "--development-seeds", default="2300,2301,2302,2303,2304"
    )
    parser.add_argument("--environment-profile")
    parser.add_argument("--audit-seed", type=int, default=2299)
    parser.add_argument("--random-permutations", type=int, default=20)
    parser.add_argument("--targeted-swaps", type=int, default=10)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_agent(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config_data = dict(payload["config"])
    config_data["precision"] = "fp32"
    config = BranchingAgentConfig(**config_data)
    if config.architecture != "equivariant_set_branching":
        raise ValueError("audit requires the equivariant set architecture")
    if config.state_schema != PHASE2D_POLICY_SCHEMA:
        raise ValueError("audit requires the Phase 2D observation schema")
    agent = BranchingDQNAgent(config, device="cpu")
    agent.online.load_state_dict(payload["online_state_dict"])
    agent.online.eval()
    return agent, payload


def network_outputs(agent, state, mask):
    state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
    mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
    with torch.no_grad():
        transformed = agent._transform_state_tensor(state_t)
        log_probabilities = agent.online(transformed, mask_t)[0].cpu().numpy()
    q_values = (
        np.exp(log_probabilities)
        * agent.online.support.detach().cpu().numpy()[None, None, :]
    ).sum(axis=-1)
    return log_probabilities, q_values


def compare(agent, state, mask, caps, priorities, permutation, kind, index):
    reference_log, reference_q = network_outputs(agent, state, mask)
    reference_action = agent._project(
        reference_q,
        mask,
        caps=caps,
        tie_break_priorities=priorities,
    )
    bundle = permute_complete_bundle(
        state,
        mask,
        caps,
        action_mask_from_caps(mask, caps, agent.cfg.actions),
        permutation,
        tie_break_priorities=priorities,
    )
    moved_log, moved_q = network_outputs(
        agent, bundle["state"], bundle["mask"]
    )
    moved_action = agent._project(
        moved_q,
        bundle["mask"],
        caps=bundle["caps"],
        tie_break_priorities=bundle["tie_break_priorities"],
    )
    physical_log = inverse_map_branch_values(moved_log, permutation)
    physical_q = inverse_map_branch_values(moved_q, permutation)
    physical_action = inverse_map_branch_values(moved_action, permutation)
    active = np.asarray(mask, dtype=bool)
    local_reference = np.argmax(reference_q, axis=1)
    local_physical = np.argmax(physical_q, axis=1)
    return {
        "kind": kind,
        "index": int(index),
        "max_abs_log_probability_error": float(
            np.max(np.abs(physical_log[active] - reference_log[active]))
        ),
        "max_abs_q_error": float(
            np.max(np.abs(physical_q[active] - reference_q[active]))
        ),
        "local_argmax_agreement": float(
            np.mean(local_physical[active] == local_reference[active])
        ),
        "projected_allocation_agreement": float(
            np.mean(physical_action == reference_action)
        ),
        "projected_allocation_l1": int(
            np.abs(physical_action - reference_action).sum()
        ),
        "budget_feasible": bool(
            physical_action.sum() <= agent.cfg.budget
        ),
        "caps_feasible": bool(np.all(physical_action <= caps)),
    }


def summarize(records):
    return {
        "count": len(records),
        "maximum_log_probability_error": max(
            row["max_abs_log_probability_error"] for row in records
        ),
        "maximum_q_error": max(row["max_abs_q_error"] for row in records),
        "mean_local_argmax_agreement": float(
            np.mean([row["local_argmax_agreement"] for row in records])
        ),
        "mean_projected_allocation_agreement": float(
            np.mean(
                [row["projected_allocation_agreement"] for row in records]
            )
        ),
        "maximum_projected_allocation_l1": max(
            row["projected_allocation_l1"] for row in records
        ),
        "all_allocations_feasible": all(
            row["budget_feasible"] and row["caps_feasible"]
            for row in records
        ),
    }


def main():
    args = parse_args()
    development_seeds = [
        int(value)
        for value in args.development_seeds.split(",")
        if value.strip()
    ]
    if not development_seeds:
        raise ValueError("at least one development seed is required")
    checkpoint = args.checkpoint.resolve()
    agent, payload = load_agent(checkpoint)
    environments, manifest, _ = build_curriculum(
        development_seeds,
        args.max_steps,
        observation_schema=agent.cfg.state_schema,
        environment_profile=args.environment_profile,
    )
    rng = np.random.default_rng(args.audit_seed)
    random_records = []
    targeted_records = []
    boundary_mass = []
    q_values_all = []

    for environment in environments:
        observation, mask, _ = environment.reset()
        state, active, caps = padded_state(
            environment, observation, mask, environment.base.n_nodes
        )
        if np.count_nonzero(active) < 2:
            continue
        priorities = np.arange(environment.base.n_nodes, dtype=np.int64)
        log_probabilities, q_values = network_outputs(agent, state, active)
        probabilities = np.exp(log_probabilities[active])
        boundary_mass.extend(
            (probabilities[..., 0] + probabilities[..., -1]).ravel().tolist()
        )
        q_values_all.extend(q_values[active].ravel().tolist())

        if len(random_records) < args.random_permutations:
            order = active_branch_permutation(active, rng)
            random_records.append(
                compare(
                    agent,
                    state,
                    active,
                    caps,
                    priorities,
                    order,
                    "random",
                    len(random_records),
                )
            )
        if len(targeted_records) < args.targeted_swaps:
            active_nodes = np.flatnonzero(active)
            if active_nodes.size >= 2:
                first = int(active_nodes[0])
                second = int(active_nodes[-1])
                order = swap_permutation(len(active), first, second)
                targeted_records.append(
                    compare(
                        agent,
                        state,
                        active,
                        caps,
                        priorities,
                        order,
                        "targeted_swap",
                        len(targeted_records),
                    )
                )
        if (
            len(random_records) >= args.random_permutations
            and len(targeted_records) >= args.targeted_swaps
        ):
            break

    if len(random_records) != args.random_permutations:
        raise RuntimeError("insufficient random permutation probes")
    if len(targeted_records) != args.targeted_swaps:
        raise RuntimeError("insufficient targeted swap probes")

    random_summary = summarize(random_records)
    targeted_summary = summarize(targeted_records)
    boundary = np.asarray(boundary_mass, dtype=np.float64)
    q_array = np.asarray(q_values_all, dtype=np.float64)
    atom_spacing = (agent.cfg.v_max - agent.cfg.v_min) / (agent.cfg.atoms - 1)
    near_boundary = (q_array <= agent.cfg.v_min + atom_spacing) | (
        q_array >= agent.cfg.v_max - atom_spacing
    )
    categorical = {
        "sample_count": int(boundary.size),
        "median_boundary_atom_mass": float(np.median(boundary)),
        "boundary_atom_mass_iqr": [
            float(np.quantile(boundary, 0.25)),
            float(np.quantile(boundary, 0.75)),
        ],
        "q_value_count": int(q_array.size),
        "q_within_one_atom_of_boundary_fraction": float(
            np.mean(near_boundary)
        ),
        "q_quantiles": {
            "q0_01": float(np.quantile(q_array, 0.01)),
            "q0_50": float(np.quantile(q_array, 0.50)),
            "q0_99": float(np.quantile(q_array, 0.99)),
        },
    }
    gates = {
        "random_log_probability_error_le_1e_6": (
            random_summary["maximum_log_probability_error"] <= 1e-6
        ),
        "random_q_error_le_1e_6": random_summary["maximum_q_error"] <= 1e-6,
        "random_allocation_agreement_ge_0_95": (
            random_summary["mean_projected_allocation_agreement"] >= 0.95
        ),
        "targeted_allocation_agreement_ge_0_90": (
            targeted_summary["mean_projected_allocation_agreement"] >= 0.90
        ),
        "local_argmax_agreement_is_one": (
            random_summary["mean_local_argmax_agreement"] == 1.0
            and targeted_summary["mean_local_argmax_agreement"] == 1.0
        ),
        "all_allocations_feasible": (
            random_summary["all_allocations_feasible"]
            and targeted_summary["all_allocations_feasible"]
        ),
        "median_boundary_atom_mass_lt_0_10": (
            categorical["median_boundary_atom_mass"] < 0.10
        ),
        "q_near_boundary_fraction_le_0_05": (
            categorical["q_within_one_atom_of_boundary_fraction"] <= 0.05
        ),
    }
    report = {
        "status": "gate_pass" if all(gates.values()) else "gate_fail",
        "scope": "development-only Phase 2D learning-smoke audit",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_metadata": payload.get("metadata", {}),
        "agent_config": payload["config"],
        "online_parameter_count": sum(
            parameter.numel() for parameter in agent.online.parameters()
        ),
        "development_seeds": development_seeds,
        "environment_profile": (
            manifest[0].get("environment_profile") if manifest else None
        ),
        "curriculum_pair_count": len(manifest),
        "random_permutations": random_summary,
        "targeted_swaps": targeted_summary,
        "categorical_boundary": categorical,
        "gates": gates,
        "records": random_records + targeted_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **gates}, indent=2))
    print(f"report={args.output.resolve()}")
    return 0 if report["status"] == "gate_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
