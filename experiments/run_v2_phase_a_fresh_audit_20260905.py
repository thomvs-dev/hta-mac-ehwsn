"""CPU-parallel V2 Phase-A marginal service/energy mechanism audit.

This is evaluation and instrumentation only.  It never updates a model.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path

# Set native library limits before Windows workers import NumPy/Torch.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.train_phase2_dynamic_curriculum as trainer
from agents.qos_constraints_v3 import Step3QoSConstraintConfig
from envs.step3_policy_observation import STEP3_CH_CONTEXT_SCHEMA
from experiments.diagnose_step3_delivery_feasibility import load_agent
from experiments.distill_step3_qos_shield import set_cpu_contract
from experiments.evaluate_step4_publication_evidence import (
    OnlinePrimalDualQoS,
    build_transfer_environments,
)
from experiments.run_step4_node_scalability_extension import (
    capped_energy_proportional_action,
)


POLICIES = ("hta_mac_v1", "residual_energy_cap_corrected", "online_primal_dual")
ROLE_FIELDS = ("member_tx", "ch_rx", "ch_aggregate", "ch_tx_bs", "idle")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(value: str | Path) -> str:
    digest = hashlib.sha256()
    with resolve(value).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_values(value, key: str = "") -> set[int]:
    found: set[int] = set()
    if isinstance(value, dict):
        for child_key, child in value.items():
            found.update(_seed_values(child, str(child_key)))
    elif isinstance(value, list):
        for child in value:
            found.update(_seed_values(child, key))
    elif isinstance(value, int) and not isinstance(value, bool) and "seed" in key.lower():
        found.add(int(value))
    return found


def seed_inventory(contract: dict, contract_path: Path) -> dict:
    excluded = {resolve(path) for path in contract["seed_registry_exclude"]}
    files: set[Path] = set()
    for pattern in contract["seed_registry_globs"]:
        files.update(ROOT.glob(pattern))
    records = []
    all_seeds: set[int] = set()
    for path in sorted(files):
        if path.resolve() in excluded or path.resolve() == contract_path.resolve():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        values = sorted(_seed_values(payload))
        if values:
            all_seeds.update(values)
            records.append({"path": str(path.relative_to(ROOT)), "seeds": values})
    development = set(map(int, contract["development_seeds"]))
    opened = set(map(int, contract["permanently_opened_confirmation_seeds"]))
    return {
        "registry_files_with_seeds": records,
        "registry_seed_count": len(all_seeds),
        "registry_seeds": sorted(all_seeds),
        "development_seeds": sorted(development),
        "development_overlap_registry": sorted(development & all_seeds),
        "development_overlap_opened_3900_3919": sorted(development & opened),
        "fresh_development_cohort": not bool(development & (all_seeds | opened)),
    }


def load_contract(path: Path) -> tuple[dict, dict]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("status") != "frozen_before_v2_phase_a_mechanism_audit":
        raise RuntimeError("Phase-A contract is not frozen")
    if contract.get("phase") != "A_only_no_training":
        raise RuntimeError("only Phase A is authorized")
    if tuple(contract["policies"]) != POLICIES:
        raise RuntimeError("Phase-A policy set changed")
    if contract["scenario"].get("id") != "reference_100":
        raise RuntimeError("Gate A must use the frozen reference scenario")
    if int(contract["horizon"]) != 3000 or int(contract["budget"]) != 24:
        raise RuntimeError("frozen horizon/budget changed")
    if float(contract["attribution_contract"]["threshold"]) != 0.8:
        raise RuntimeError("Gate A threshold must equal 0.80 exactly")
    if int(contract["workers"]) * int(contract["threads_per_worker"]) > 16:
        raise RuntimeError("CPU contract exceeds the preregistered 16-worker limit")
    for field in ("source_contract", "source_checkpoint", "qos_config", "risk_config"):
        if sha256(contract[field]) != contract[f"{field}_sha256"]:
            raise RuntimeError(f"artifact checksum mismatch: {field}")
    source = json.loads(resolve(contract["source_contract"]).read_text(encoding="utf-8"))
    if contract["primal_dual"] != source["primal_dual"]:
        raise RuntimeError("primal-dual comparator parameters changed")
    inventory = seed_inventory(contract, path)
    if not inventory["fresh_development_cohort"]:
        raise RuntimeError("Phase-A development seeds overlap prior/opened seed evidence")
    return contract, inventory


def shapley_efficiency_decomposition(
    packets_hta: float, energy_hta: float, packets_reference: float, energy_reference: float
) -> dict:
    values = (packets_hta, energy_hta, packets_reference, energy_reference)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("efficiency inputs must be finite")
    if energy_hta <= 0.0 or energy_reference <= 0.0:
        raise ValueError("efficiency energy denominators must be positive")
    observed = packets_reference / energy_reference - packets_hta / energy_hta
    service = 0.5 * (
        (packets_reference / energy_hta - packets_hta / energy_hta)
        + (packets_reference / energy_reference - packets_hta / energy_reference)
    )
    energy = 0.5 * (
        (packets_hta / energy_reference - packets_hta / energy_hta)
        + (packets_reference / energy_reference - packets_reference / energy_hta)
    )
    if not np.isclose(service + energy, observed, rtol=0.0, atol=1e-10):
        raise RuntimeError("efficiency Shapley decomposition is not exact")
    return {
        "observed_gap_packets_per_j": float(observed),
        "service_component_packets_per_j": float(service),
        "energy_component_packets_per_j": float(energy),
        "reconstruction_error": float(service + energy - observed),
    }


def gate_a_attribution(aggregates: dict, threshold: float) -> dict:
    hta = aggregates["hta_mac_v1"]
    reference = aggregates["residual_energy_cap_corrected"]
    delivery_gap = max(0.0, reference["mean_delivery_ratio"] - hta["mean_delivery_ratio"])
    capacity_delta = max(
        0.0,
        hta["mean_under_service_capacity_ratio"]
        - reference["mean_under_service_capacity_ratio"],
    )
    delivery_attributed = min(delivery_gap, capacity_delta)
    delivery_fraction = 1.0 if delivery_gap <= 0.0 else delivery_attributed / delivery_gap
    decomposition = shapley_efficiency_decomposition(
        hta["mean_global_packets"], hta["mean_network_energy_j"],
        reference["mean_global_packets"], reference["mean_network_energy_j"],
    )
    efficiency_gap = max(0.0, decomposition["observed_gap_packets_per_j"])
    service_positive = max(0.0, decomposition["service_component_packets_per_j"])
    energy_positive = max(0.0, decomposition["energy_component_packets_per_j"])
    service_attributed = service_positive * min(1.0, delivery_fraction)
    role_reconstruction = bool(
        hta["max_role_reconstruction_error_j"] <= 1e-12
        and reference["max_role_reconstruction_error_j"] <= 1e-12
    )
    favorable_energy = reference["mean_network_energy_j"] < hta["mean_network_energy_j"]
    energy_attributed = energy_positive if role_reconstruction and favorable_energy else 0.0
    efficiency_attributed = min(efficiency_gap, service_attributed + energy_attributed)
    efficiency_fraction = 1.0 if efficiency_gap <= 0.0 else efficiency_attributed / efficiency_gap
    minimum_fraction = min(delivery_fraction, efficiency_fraction)
    return {
        "threshold": float(threshold),
        "delivery": {
            "observed_positive_gap": float(delivery_gap),
            "hta_minus_reference_under_service_capacity_ratio": float(capacity_delta),
            "attributed_gap": float(delivery_attributed),
            "attribution_fraction": float(delivery_fraction),
        },
        "efficiency": {
            **decomposition,
            "observed_positive_gap": float(efficiency_gap),
            "service_attributed": float(service_attributed),
            "energy_attributed": float(energy_attributed),
            "attributed_gap": float(efficiency_attributed),
            "attribution_fraction": float(efficiency_fraction),
            "role_energy_reconstruction_within_1e_12_j": role_reconstruction,
            "reference_uses_less_energy": favorable_energy,
        },
        "minimum_attribution_fraction": float(minimum_fraction),
        "gate_a_pass": bool(minimum_fraction >= float(threshold)),
        "continuation_authorized": bool(minimum_fraction >= float(threshold)),
    }


def _allocated_slot_keys(action: np.ndarray) -> set[tuple[int, int]]:
    return {
        (int(node), layer)
        for node in np.flatnonzero(action > 0)
        for layer in range(1, int(action[node]) + 1)
    }


def controlled_energy(action, packet_costs, forwarding, idle_per_slot):
    """Exact target-cluster consumption, including shared setup and idle cost."""
    slots = int(np.sum(action))
    participants = int(np.count_nonzero(action))
    return float(np.dot(action, packet_costs) + (forwarding if slots else 0.0)
                 + max(0, participants - 1) * slots * idle_per_slot)


def validate_action(action, active, caps, budget):
    action = np.asarray(action)
    if (not np.all(np.isfinite(action)) or np.any(action != np.floor(action))
            or np.any(action < 0) or np.any(action > caps)
            or np.any(action[~np.asarray(active, dtype=bool)] != 0)
            or action.sum() > budget):
        raise RuntimeError("Phase-A budget/cap/alive-mask feasibility failure")


def _slot_candidates(env, caps, action, queue_before, ages_before, info, energy_before, positions_before):
    current_ch = int(info["target_ch"])
    current_members = np.asarray(info["target_members"], dtype=np.int64)
    role = info["energy_trace"]["role_energy"]
    delivered = np.asarray(info["target_packets_delivered_per_node"], dtype=np.int64)
    allocated_keys = _allocated_slot_keys(action)
    first_allocated = min(allocated_keys) if allocated_keys else None
    ch_forwarding = float(role["ch_tx_bs"][current_ch])
    costs = np.zeros_like(action, dtype=float)
    for node in current_members:
        distance = float(np.linalg.norm(positions_before[node] - positions_before[current_ch]))
        costs[node] = (env.base.radio.tx(env.base.cfg.packet_bits, distance)
                       + env.base.radio.rx(env.base.cfg.packet_bits)
                       + env.base.radio.aggregate(env.base.cfg.packet_bits))
    forwarding_setup = float(env.base.radio.tx(env.base.cfg.packet_bits,
        float(np.linalg.norm(positions_before[current_ch] - np.asarray(env.base.cfg.bs_position_m)))))
    idle_per_slot = (float(env.base.cfg.e_elec_j_per_bit * env.base.cfg.idle_slot_bit_times)
                     if env.base.idle_energy_enabled else 0.0)
    actual_target_energy = float(np.asarray(role["member_tx"])[current_members].sum()
        + role["ch_rx"][current_ch] + role["ch_aggregate"][current_ch]
        + role["ch_tx_bs"][current_ch] + np.asarray(role["idle"])[current_members].sum())
    if abs(controlled_energy(action, costs, forwarding_setup, idle_per_slot) - actual_target_energy) > 1e-12:
        raise RuntimeError("Pre-step target slot/idle energy does not reconstruct simulator consumption")
    result = []
    for node in current_members:
        distance = float(np.linalg.norm(positions_before[node] - positions_before[current_ch]))
        member_packet_energy = float(env.base.radio.tx(env.base.cfg.packet_bits, distance))
        ch_rx_packet_energy = float(env.base.radio.rx(env.base.cfg.packet_bits))
        ch_agg_packet_energy = float(env.base.radio.aggregate(env.base.cfg.packet_bits))
        for layer in range(1, int(caps[node]) + 1):
            key = (int(node), layer)
            allocated = key in allocated_keys
            success = bool(allocated and layer <= int(delivered[node]))
            forwarding = ch_forwarding if success and key == first_allocated else 0.0
            marginal_energy = (
                member_packet_energy + ch_rx_packet_energy + ch_agg_packet_energy + forwarding
                if success else 0.0
            )
            packet_age = int(ages_before[node][layer - 1])
            lower = np.asarray(action).copy()
            upper = lower.copy()
            lower[node], upper[node] = layer - 1, layer
            exact_increment = (controlled_energy(upper, costs, forwarding_setup, idle_per_slot)
                               - controlled_energy(lower, costs, forwarding_setup, idle_per_slot))
            result.append({
                "node": int(node), "slot_layer": layer, "feasible": True,
                "allocated": allocated, "real_queue_service": bool(success),
                "delivery_success": bool(success),
                "packet_age_before": packet_age,
                "stale_avoidance": bool(success and packet_age >= env.base.cfg.packet_ttl_rounds),
                "member_tx_energy_j": member_packet_energy if success else 0.0,
                "ch_receive_energy_j": ch_rx_packet_energy if success else 0.0,
                "ch_aggregation_energy_j": ch_agg_packet_energy if success else 0.0,
                "ch_forwarding_energy_j": forwarding,
                "marginal_energy_j": marginal_energy,
                "marginal_packets_per_j": 1.0 / marginal_energy if marginal_energy > 0.0 else None,
                "member_energy_before_j": float(energy_before[node]),
                "scheduled_ch_energy_before_j": float(energy_before[current_ch]),
                "counterfactual_incremental_energy_including_idle_j": exact_increment,
                "counterfactual_incremental_packets_per_j": 1.0 / exact_increment if exact_increment > 0 else None,
                "counterfactual_upper_action_budget_feasible": bool(upper.sum() <= env.base.cfg.frame_slot_budget),
                "counterfactual_is_next_slot_from_actual_action": bool(layer == action[node] + 1),
            })
    return result


def evaluate_task(contract_path: str, policy: str, seed: int, trace_dir: str):
    contract, _ = load_contract(resolve(contract_path))
    set_cpu_contract(int(contract["threads_per_worker"]), int(seed) + len(policy))
    runtime = dict(contract)
    runtime["development_seeds"] = [int(seed)]
    runtime["observation_schema"] = STEP3_CH_CONTEXT_SCHEMA
    environments = build_transfer_environments(runtime, contract["scenario"], trace=None)
    qos = Step3QoSConstraintConfig.from_payload(
        json.loads(resolve(contract["qos_config"]).read_text(encoding="utf-8"))
    )
    agent = load_agent(resolve(contract["source_checkpoint"]))[0] if policy == "hta_mac_v1" else None
    trace_path = Path(trace_dir) / f"{policy}_seed_{seed}.jsonl.gz"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    episode_rows = []
    decision_count = candidate_count = 0
    with trace_path.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as gz_handle:
            with io.TextIOWrapper(gz_handle, encoding="utf-8", newline="\n") as trace_handle:
                for env in environments:
                    observation, mask, _ = env.reset()
                    dual = OnlinePrimalDualQoS(qos, contract["primal_dual"], int(contract["budget"]))
                    done = False
                    totals = {
                        "allocated_slots": 0, "real_queue_service_slots": 0,
                        "stale_avoidance_slots": 0, "unused_budget": 0,
                        "unallocated_feasible_demand": 0, "under_service_capacity": 0,
                        "selection_energy_regret_j": 0.0,
                    }
                    role_totals = {name: 0.0 for name in ROLE_FIELDS}
                    max_reconstruction_error = 0.0
                    while not done:
                        state, active, caps = trainer.padded_state(
                            env, observation, mask, env.base.n_nodes
                        )
                        if policy == "hta_mac_v1":
                            action, _ = agent.act(
                                state, active, epsilon=0.0, caps=caps,
                                budget=int(contract["budget"]),
                            )
                        elif policy == "residual_energy_cap_corrected":
                            action = capped_energy_proportional_action(
                                env, active, caps, int(contract["budget"]),
                                float(contract["energy_proportional_score_exponent"]),
                            )
                        elif policy == "online_primal_dual":
                            action = dual.action(env, active, caps)
                        else:
                            raise ValueError(policy)
                        validate_action(action, active, caps, int(contract["budget"]))
                        queue_before = env.base.queue.copy()
                        positions_before = env.base.positions.copy()
                        ages_before = [list(ages) for ages in env.base.packet_ages]
                        energy_before = env.base.energy.copy()
                        round_before = int(env.base.round)
                        allocated = int(action.sum())
                        feasible = int(caps.sum())
                        unused_budget = max(0, int(contract["budget"]) - allocated)
                        unallocated = max(0, feasible - allocated)
                        under_service = min(unused_budget, unallocated)
                        observation, mask, done, info = env.step(action)
                        candidates = _slot_candidates(
                            env, caps, action, queue_before, ages_before, info, energy_before, positions_before
                        )
                        # The one-per-frame CH forwarding setup is common to every
                        # non-empty selection, so exclude it from ranking regret.
                        allocated_costs = [
                            row["member_tx_energy_j"]
                            + row["ch_receive_energy_j"]
                            + row["ch_aggregation_energy_j"]
                            for row in candidates if row["allocated"]
                        ]
                        feasible_costs = sorted(
                            env.base.radio.tx(
                                env.base.cfg.packet_bits,
                                float(np.linalg.norm(positions_before[row["node"]] - positions_before[int(info["target_ch"])])),
                            )
                            + env.base.radio.rx(env.base.cfg.packet_bits)
                            + env.base.radio.aggregate(env.base.cfg.packet_bits)
                            for row in candidates
                        )
                        regret = max(0.0, sum(allocated_costs) - sum(feasible_costs[:len(allocated_costs)]))
                        role = info["energy_trace"]["role_energy"]
                        consumed = float(np.asarray(info["energy_trace"]["consumed"]).sum())
                        reconstructed = 0.0
                        for name in ROLE_FIELDS:
                            value = float(np.asarray(role[name]).sum())
                            role_totals[name] += value
                            reconstructed += value
                        max_reconstruction_error = max(max_reconstruction_error, abs(consumed - reconstructed))
                        totals["allocated_slots"] += allocated
                        totals["real_queue_service_slots"] += int(info["target_packets_delivered"])
                        totals["stale_avoidance_slots"] += sum(row["stale_avoidance"] for row in candidates)
                        totals["unused_budget"] += unused_budget
                        totals["unallocated_feasible_demand"] += unallocated
                        totals["under_service_capacity"] += under_service
                        totals["selection_energy_regret_j"] += regret
                        decision = {
                            "schema_version": 1, "policy": policy, "seed": int(seed),
                            "target_rank": int(env.target_rank), "round_before": round_before,
                            "target_cluster": int(info["target_cluster"]),
                            "scheduled_ch": int(info["target_ch"]),
                            "budget": int(contract["budget"]), "feasible_slot_count": feasible,
                            "allocated_slot_count": allocated, "unused_budget": unused_budget,
                            "unallocated_feasible_demand": unallocated,
                            "under_service_capacity": under_service,
                            "selection_energy_regret_j": regret,
                            "scheduled_ch_energy_before_j": float(energy_before[int(info["target_ch"])]),
                            "scheduled_ch_energy_after_j": float(env.base.energy[int(info["target_ch"])]),
                            "role_energy_j": {name: float(np.asarray(role[name]).sum()) for name in ROLE_FIELDS},
                            "role_reconstruction_error_j": abs(consumed - reconstructed),
                            "slot_candidates": candidates,
                        }
                        trace_handle.write(json.dumps(decision, separators=(",", ":")) + "\n")
                        decision_count += 1
                        candidate_count += len(candidates)
                    counts = env.step3_qos_counts
                    demand = max(1, int(counts["demand"]))
                    consumed_total = sum(role_totals.values())
                    episode_rows.append({
                        "policy": policy, "seed": int(seed), "target_rank": int(env.target_rank),
                        "delivery_ratio": int(counts["delivered"]) / demand,
                        "stale_ratio": int(counts["stale"]) / demand,
                        "fairness": float(counts["episode_service_fairness"]),
                        "restricted_survival_rounds": int(
                            env.base.t_fnd if env.base.t_fnd is not None else contract["horizon"]
                        ),
                        "global_packets": int(env.base.total_packets),
                        "network_energy_j": float(consumed_total),
                        "packets_per_j": int(env.base.total_packets) / max(consumed_total, 1e-12),
                        "target_demand": int(demand),
                        "under_service_capacity_ratio": totals["under_service_capacity"] / demand,
                        "max_role_reconstruction_error_j": float(max_reconstruction_error),
                        "zero_feasibility_violations": bool(
                            totals["allocated_slots"] == totals["real_queue_service_slots"]
                        ),
                        "role_energy_j": role_totals,
                        **totals,
                    })
    return {
        "policy": policy, "seed": int(seed), "episodes": episode_rows,
        "trace": str(trace_path), "trace_sha256": sha256(trace_path),
        "decision_rows": decision_count, "slot_candidate_rows": candidate_count,
    }


def summarize_seed(episodes: list[dict]) -> dict:
    scalar_fields = (
        "delivery_ratio", "stale_ratio", "fairness", "restricted_survival_rounds",
        "global_packets", "network_energy_j", "packets_per_j", "target_demand",
        "under_service_capacity_ratio", "allocated_slots", "real_queue_service_slots",
        "stale_avoidance_slots", "unused_budget", "unallocated_feasible_demand",
        "under_service_capacity", "selection_energy_regret_j",
    )
    result = {f"mean_{name}": float(np.mean([row[name] for row in episodes])) for name in scalar_fields}
    result["max_role_reconstruction_error_j"] = float(
        max(row["max_role_reconstruction_error_j"] for row in episodes)
    )
    result["rank_units"] = len(episodes)
    result["zero_feasibility_violations"] = all(row["zero_feasibility_violations"] for row in episodes)
    result["mean_role_energy_j"] = {
        name: float(np.mean([row["role_energy_j"][name] for row in episodes]))
        for name in ROLE_FIELDS
    }
    return result


def aggregate_policy(seed_rows: dict[str, dict]) -> dict:
    fields = [key for key in next(iter(seed_rows.values())) if key.startswith("mean_") and key != "mean_role_energy_j"]
    result = {field: float(np.mean([row[field] for row in seed_rows.values()])) for field in fields}
    result["max_role_reconstruction_error_j"] = float(
        max(row["max_role_reconstruction_error_j"] for row in seed_rows.values())
    )
    result["seed_units"] = len(seed_rows)
    result["zero_feasibility_violations"] = all(row["zero_feasibility_violations"] for row in seed_rows.values())
    result["mean_role_energy_j"] = {
        name: float(np.mean([row["mean_role_energy_j"][name] for row in seed_rows.values()]))
        for name in ROLE_FIELDS
    }
    return result


def write_seed_csv(path: Path, seed_summaries: dict):
    rows = []
    for policy, seeds in seed_summaries.items():
        for seed, row in seeds.items():
            flat = {"policy": policy, "seed": int(seed)}
            flat.update({key: value for key, value in row.items() if not isinstance(value, dict)})
            for role, value in row["mean_role_energy_j"].items():
                flat[f"mean_role_energy_{role}_j"] = value
            rows.append(flat)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def render_report(payload: dict) -> str:
    gate = payload["gate_a"]
    lines = [
        "# HTA-MAC V2 Phase A marginal service/energy audit",
        "",
        f"**Status:** `{payload['status']}`  ",
        f"**Gate A:** `{'PASS' if gate['gate_a_pass'] else 'FAIL'}` at the frozen 0.80 threshold  ",
        f"**Branch:** `{payload['git_branch']}`  ",
        "**Scope:** CPU-only mechanism audit; MAC allocation only; HEART-CH schedule exogenous; no training.",
        "",
        "## Gate A result",
        "",
        f"- Delivery attribution fraction: `{gate['delivery']['attribution_fraction']:.6f}`.",
        f"- Packets/J attribution fraction: `{gate['efficiency']['attribution_fraction']:.6f}`.",
        f"- Minimum controlling fraction: `{gate['minimum_attribution_fraction']:.6f}`.",
        f"- Continuation authorized: `{str(gate['continuation_authorized']).lower()}`.",
        "",
        "The attribution rule was frozen before the fresh cohort ran. Delivery attribution is capped by the measured HTA-minus-reference under-service capacity. Efficiency uses an exact two-order Shapley decomposition into packet-service and joule components; the joule component is credited only when role-energy reconstruction is exact and favorable.",
        "",
        "## Cohort and integrity",
        "",
        f"- Fresh development seeds: `{payload['development_seeds'][0]}--{payload['development_seeds'][-1]}`.",
        "- Opened seeds 3900--3919 excluded: `true`.",
        f"- Seed-registry overlap: `{payload['seed_inventory']['development_overlap_registry']}`.",
        f"- Complete tasks: `{payload['checks']['all_tasks_complete']}`; exact pairing: `{payload['checks']['paired_seed_cohorts']}`; zero feasibility violations: `{payload['checks']['zero_feasibility_violations']}`.",
        "",
        "## Policy means",
        "",
        "| Policy | Delivery | Packets/J | Under-service capacity ratio | Member TX J | CH RX J | CH aggregate J | CH forward J |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        row = payload["aggregates"][policy]
        role = row["mean_role_energy_j"]
        lines.append(
            f"| {policy} | {row['mean_delivery_ratio']:.6f} | {row['mean_packets_per_j']:.6f} | "
            f"{row['mean_under_service_capacity_ratio']:.6f} | {role['member_tx']:.6f} | "
            f"{role['ch_rx']:.6f} | {role['ch_aggregate']:.6f} | {role['ch_tx_bs']:.6f} |"
        )
    lines.extend([
        "", "## Evidence boundary", "",
        "Decision traces are gzip-compressed JSON Lines. Every line records a decision and every feasible slot candidate, including allocation/service outcome, stale avoidance, marginal role energy, member energy, and scheduled-CH energy. These are development diagnostics, not confirmation evidence.",
        "",
        "No Phase B teacher, architecture work, imitation, PPO, C51 fine-tuning, or other neural training was started.",
    ])
    if not gate["gate_a_pass"]:
        lines.extend(["", "Gate A failed. Work stops here; the threshold and failure are preserved."])
    else:
        lines.extend(["", "Gate A passed. This report still stops at Phase A; later phases require a separate authorized task."])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, help="Capacity-only reduction; never exceeds frozen maximum")
    args = parser.parse_args()
    contract_path = resolve(args.contract)
    output_dir = resolve(args.output_dir)
    report_path = resolve(args.report)
    contract, inventory = load_contract(contract_path)
    workers = args.workers if args.workers is not None else int(contract["workers"])
    if not 1 <= workers <= int(contract["workers"]):
        raise ValueError("Worker override must be between one and the frozen maximum")
    if (output_dir / "results.json").exists() or (output_dir / "decision_traces").exists() or report_path.exists():
        raise FileExistsError("Refusing to overwrite prior evidence; choose a fresh output/report path")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "seed_inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
    )
    started = time.perf_counter()
    tasks = [(policy, int(seed)) for policy in POLICIES for seed in contract["development_seeds"]]
    completed = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(evaluate_task, str(contract_path), policy, seed, str(output_dir / "decision_traces")): (policy, seed)
            for policy, seed in tasks
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            completed.append(future.result())
            if index % 5 == 0 or index == len(tasks):
                print(f"PHASE_A_PROGRESS={index}/{len(tasks)}", flush=True)
    seed_summaries = {policy: {} for policy in POLICIES}
    traces = []
    for task in completed:
        seed_summaries[task["policy"]][str(task["seed"])] = summarize_seed(task["episodes"])
        traces.append({key: task[key] for key in ("policy", "seed", "trace", "trace_sha256", "decision_rows", "slot_candidate_rows")})
    aggregates = {policy: aggregate_policy(seed_summaries[policy]) for policy in POLICIES}
    expected_seeds = set(map(str, contract["development_seeds"]))
    checks = {
        "fresh_development_cohort": inventory["fresh_development_cohort"],
        "opened_3900_3919_excluded": not inventory["development_overlap_opened_3900_3919"],
        "all_tasks_complete": len(completed) == len(tasks),
        "paired_seed_cohorts": all(set(seed_summaries[policy]) == expected_seeds for policy in POLICIES),
        "zero_feasibility_violations": all(aggregates[policy]["zero_feasibility_violations"] for policy in POLICIES),
        "role_energy_reconstruction": all(aggregates[policy]["max_role_reconstruction_error_j"] <= 1e-12 for policy in POLICIES),
        "no_training": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase-A integrity failure: {checks}")
    gate = gate_a_attribution(aggregates, contract["attribution_contract"]["threshold"])
    import subprocess
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    payload = {
        "schema_version": 1,
        "status": "phase_a_gate_pass" if gate["gate_a_pass"] else "phase_a_gate_fail_stop",
        "phase": "A_only_no_training", "git_branch": branch,
        "contract": str(contract_path), "contract_sha256": sha256(contract_path),
        "runner_sha256": sha256(Path(__file__)),
        "source_checkpoint_sha256": sha256(contract["source_checkpoint"]),
        "development_seeds": contract["development_seeds"],
        "seed_inventory": inventory, "checks": checks,
        "seed_summaries": seed_summaries, "aggregates": aggregates,
        "gate_a": gate, "decision_traces": sorted(traces, key=lambda row: (row["policy"], row["seed"])),
        "elapsed_seconds": time.perf_counter() - started,
        "runtime_workers": workers,
        "threads_per_worker": int(contract["threads_per_worker"]),
        "selection_or_retuning_performed": False,
        "neural_training_performed": False,
        "next_phase_started": False,
        "claim_boundary": "fresh-seed Phase-A development mechanism audit in the simulated reference transfer environment",
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_seed_csv(output_dir / "seed_level_metrics.csv", seed_summaries)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"], "gate_a": gate,
        "elapsed_seconds": payload["elapsed_seconds"],
        "results": str(results_path), "report": str(report_path),
    }, indent=2), flush=True)
    return 0 if gate["gate_a_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
