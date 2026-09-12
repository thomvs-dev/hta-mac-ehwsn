"""Factorial causal replay for the failed HTA-MAC V2 Phase-A audit."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import io
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.train_phase2_dynamic_curriculum as trainer
from agents.qos_constraints_v3 import Step3QoSConstraintConfig
from envs.step3_policy_observation import STEP3_CH_CONTEXT_SCHEMA
from experiments.diagnose_step3_delivery_feasibility import load_agent
from experiments.distill_step3_qos_shield import set_cpu_contract
from experiments.evaluate_step4_publication_evidence import build_transfer_environments
from experiments.run_step4_node_scalability_extension import capped_energy_proportional_action


ARMS = (
    "hta_count_hta_selection",
    "hta_count_residual_selection",
    "full_count_hta_selection",
    "full_count_residual_selection",
)
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


def load_contract(path: Path) -> tuple[dict, dict]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("status") != "frozen_before_v2_phase_a2_causal_accounting_repair":
        raise RuntimeError("Phase A2 contract is not frozen")
    if contract.get("phase") != "A2_only_no_training":
        raise RuntimeError("only Phase A2 is authorized")
    if tuple(contract["factorial_arms"]) != ARMS:
        raise RuntimeError("factorial arms or ordering changed")
    if float(contract["gate_a"]["threshold"]) != 0.8:
        raise RuntimeError("Gate A threshold must remain exactly 0.80")
    if contract["gate_a"]["metrics"] != ["delivery_ratio", "packets_per_j"]:
        raise RuntimeError("Gate A metric family changed")
    if set(contract["development_seeds"]) & set(contract["permanently_opened_confirmation_seeds"]):
        raise RuntimeError("A2 cohort overlaps opened confirmation seeds")
    if int(contract["workers"]) * int(contract["threads_per_worker"]) > 8:
        raise RuntimeError("A2 exceeds the stable host worker contract")
    for field in ("source_phase_a_results", "source_checkpoint", "qos_config", "risk_config"):
        if sha256(contract[field]) != contract[f"{field}_sha256"]:
            raise RuntimeError(f"artifact checksum mismatch: {field}")
    phase_a = json.loads(resolve(contract["source_phase_a_results"]).read_text(encoding="utf-8"))
    if phase_a["development_seeds"] != contract["development_seeds"]:
        raise RuntimeError("A2 must reuse the opened A1 development cohort exactly")
    if phase_a["gate_a"]["gate_a_pass"] is not False:
        raise RuntimeError("A2 source must preserve the failed A1 gate")
    return contract, phase_a


def _arm_action(agent, env, state, active, caps, arm: str, budget: int, exponent: float):
    hta_action, q_values = agent.act(
        state, active, epsilon=0.0, caps=caps, budget=budget
    )
    hta_count = int(hta_action.sum())
    full_count = min(int(budget), int(caps.sum()))
    count = hta_count if arm.startswith("hta_count_") else full_count
    if arm.endswith("hta_selection"):
        if arm == "hta_count_hta_selection":
            action = hta_action
        else:
            action = agent._project(
                q_values, active, fill_budget=True, caps=caps, budget=count
            )
    elif arm.endswith("residual_selection"):
        action = capped_energy_proportional_action(
            env, active, caps, count, exponent
        )
    else:
        raise ValueError(arm)
    residual_full = capped_energy_proportional_action(
        env, active, caps, budget, exponent
    )
    return action, hta_action, residual_full, hta_count, full_count


def evaluate_seed(contract_path: str, seed: int, trace_dir: str) -> dict:
    contract, _ = load_contract(resolve(contract_path))
    set_cpu_contract(int(contract["threads_per_worker"]), int(seed) + 120_001)
    runtime = dict(contract)
    runtime["development_seeds"] = [int(seed)]
    runtime["observation_schema"] = STEP3_CH_CONTEXT_SCHEMA
    qos = Step3QoSConstraintConfig.from_payload(
        json.loads(resolve(contract["qos_config"]).read_text(encoding="utf-8"))
    )
    agent = load_agent(resolve(contract["source_checkpoint"]))[0]
    envs_by_arm = {
        arm: build_transfer_environments(runtime, contract["scenario"], trace=None)
        for arm in ARMS
    }
    trace_path = Path(trace_dir) / f"factorial_seed_{seed}.jsonl.gz"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    episodes = {arm: [] for arm in ARMS}
    action_replication_failures = 0
    decision_rows = 0
    with trace_path.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as gz_handle:
            with io.TextIOWrapper(gz_handle, encoding="utf-8", newline="\n") as handle:
                for rank_index in range(len(envs_by_arm[ARMS[0]])):
                    for arm in ARMS:
                        env = envs_by_arm[arm][rank_index]
                        observation, mask, _ = env.reset()
                        done = False
                        allocated = delivered_target = stale_avoided = 0
                        role_totals = {name: 0.0 for name in ROLE_FIELDS}
                        max_role_error = 0.0
                        selection_disagreements = count_interventions = 0
                        while not done:
                            state, active, caps = trainer.padded_state(
                                env, observation, mask, env.base.n_nodes
                            )
                            action, hta_action, residual_full, hta_count, full_count = _arm_action(
                                agent, env, state, active, caps, arm,
                                int(contract["budget"]),
                                float(contract["energy_proportional_score_exponent"]),
                            )
                            if arm == "hta_count_hta_selection" and not np.array_equal(action, hta_action):
                                action_replication_failures += 1
                            if arm == "full_count_residual_selection" and not np.array_equal(action, residual_full):
                                action_replication_failures += 1
                            if int(action.sum()) != (hta_count if arm.startswith("hta_count_") else full_count):
                                action_replication_failures += 1
                            selection_disagreements += int(not np.array_equal(action, hta_action))
                            count_interventions += int(full_count != hta_count and arm.startswith("full_count_"))
                            ages_before = [list(ages) for ages in env.base.packet_ages]
                            energy_before = env.base.energy.copy()
                            round_before = int(env.base.round)
                            observation, mask, done, info = env.step(action)
                            role = info["energy_trace"]["role_energy"]
                            consumed = float(np.asarray(info["energy_trace"]["consumed"]).sum())
                            reconstructed = 0.0
                            for name in ROLE_FIELDS:
                                value = float(np.asarray(role[name]).sum())
                                role_totals[name] += value
                                reconstructed += value
                            max_role_error = max(max_role_error, abs(consumed - reconstructed))
                            target_members = np.asarray(info["target_members"], dtype=np.int64)
                            delivered_per_node = np.asarray(
                                info["target_packets_delivered_per_node"], dtype=np.int64
                            )
                            stale_step = 0
                            for node in target_members:
                                for layer in range(int(delivered_per_node[node])):
                                    stale_step += int(
                                        ages_before[node][layer] >= env.base.cfg.packet_ttl_rounds
                                    )
                            allocated += int(action.sum())
                            delivered_target += int(info["target_packets_delivered"])
                            stale_avoided += stale_step
                            sparse_action = {
                                str(int(node)): int(action[node])
                                for node in np.flatnonzero(action > 0)
                            }
                            handle.write(json.dumps({
                                "schema_version": 1, "seed": int(seed), "arm": arm,
                                "target_rank": int(env.target_rank), "round_before": round_before,
                                "scheduled_ch": int(info["target_ch"]),
                                "hta_count": hta_count, "full_feasible_count": full_count,
                                "allocated_count": int(action.sum()),
                                "selection_differs_from_hta": bool(not np.array_equal(action, hta_action)),
                                "action": sparse_action,
                                "target_delivered": int(info["target_packets_delivered"]),
                                "stale_avoided": stale_step,
                                "scheduled_ch_energy_before_j": float(energy_before[int(info["target_ch"])]),
                                "scheduled_ch_energy_after_j": float(env.base.energy[int(info["target_ch"])]),
                                "role_energy_j": {
                                    name: float(np.asarray(role[name]).sum()) for name in ROLE_FIELDS
                                },
                                "role_reconstruction_error_j": abs(consumed - reconstructed),
                            }, separators=(",", ":")) + "\n")
                            decision_rows += 1
                        counts = env.step3_qos_counts
                        demand = max(1, int(counts["demand"]))
                        network_energy = sum(role_totals.values())
                        episodes[arm].append({
                            "arm": arm, "seed": int(seed), "target_rank": int(env.target_rank),
                            "delivery_ratio": int(counts["delivered"]) / demand,
                            "packets_per_j": int(env.base.total_packets) / max(network_energy, 1e-12),
                            "stale_ratio": int(counts["stale"]) / demand,
                            "fairness": float(counts["episode_service_fairness"]),
                            "restricted_survival_rounds": int(
                                env.base.t_fnd if env.base.t_fnd is not None else contract["horizon"]
                            ),
                            "global_packets": int(env.base.total_packets),
                            "network_energy_j": float(network_energy),
                            "allocated_slots": allocated,
                            "target_delivered_slots": delivered_target,
                            "stale_avoided_slots": stale_avoided,
                            "selection_disagreement_steps": selection_disagreements,
                            "count_intervention_steps": count_interventions,
                            "max_role_reconstruction_error_j": max_role_error,
                            "role_energy_j": role_totals,
                            "feasible": bool(allocated == delivered_target),
                        })
    return {
        "seed": int(seed), "episodes": episodes, "trace": str(trace_path),
        "trace_sha256": sha256(trace_path), "decision_rows": decision_rows,
        "action_replication_failures": action_replication_failures,
    }


def summarize_rows(rows: list[dict]) -> dict:
    fields = (
        "delivery_ratio", "packets_per_j", "stale_ratio", "fairness",
        "restricted_survival_rounds", "global_packets", "network_energy_j",
        "allocated_slots", "target_delivered_slots", "stale_avoided_slots",
        "selection_disagreement_steps", "count_intervention_steps",
    )
    return {
        **{f"mean_{field}": float(np.mean([row[field] for row in rows])) for field in fields},
        "max_role_reconstruction_error_j": float(max(row["max_role_reconstruction_error_j"] for row in rows)),
        "rank_units": len(rows), "feasible": all(row["feasible"] for row in rows),
        "mean_role_energy_j": {
            name: float(np.mean([row["role_energy_j"][name] for row in rows]))
            for name in ROLE_FIELDS
        },
    }


def aggregate_seeds(seeds: dict[str, dict]) -> dict:
    scalar = [key for key in next(iter(seeds.values())) if key.startswith("mean_") and key != "mean_role_energy_j"]
    return {
        **{key: float(np.mean([row[key] for row in seeds.values()])) for key in scalar},
        "max_role_reconstruction_error_j": float(max(row["max_role_reconstruction_error_j"] for row in seeds.values())),
        "seed_units": len(seeds), "feasible": all(row["feasible"] for row in seeds.values()),
        "mean_role_energy_j": {
            name: float(np.mean([row["mean_role_energy_j"][name] for row in seeds.values()]))
            for name in ROLE_FIELDS
        },
    }


def factorial_metric(aggregates: dict, phase_a: dict, metric: str, tolerance: float) -> dict:
    key = f"mean_{metric}"
    hh = aggregates["hta_count_hta_selection"][key]
    rh = aggregates["hta_count_residual_selection"][key]
    fh = aggregates["full_count_hta_selection"][key]
    rf = aggregates["full_count_residual_selection"][key]
    a1_hta = phase_a["aggregates"]["hta_mac_v1"][key]
    a1_reference = phase_a["aggregates"]["residual_energy_cap_corrected"][key]
    count = 0.5 * ((fh - hh) + (rf - rh))
    selection = 0.5 * ((rh - hh) + (rf - fh))
    factorial_gap = rf - hh
    reconstruction_error = count + selection - factorial_gap
    observed_gap = a1_reference - a1_hta
    unexplained = observed_gap - factorial_gap
    positive_gap = max(0.0, observed_gap)
    attributed = min(positive_gap, max(0.0, positive_gap - abs(unexplained)))
    fraction = 1.0 if positive_gap <= 0.0 else attributed / positive_gap
    return {
        "metric": metric, "a1_observed_gap": float(observed_gap),
        "a2_factorial_gap": float(factorial_gap),
        "count_rule_shapley": float(count), "selection_rule_shapley": float(selection),
        "factorial_reconstruction_error": float(reconstruction_error),
        "unexplained_accounting_residual": float(unexplained),
        "positive_observed_gap": float(positive_gap), "attributed_gap": float(attributed),
        "attribution_fraction": float(fraction),
        "hta_endpoint_replication_error": float(hh - a1_hta),
        "reference_endpoint_replication_error": float(rf - a1_reference),
        "factorial_reconstruction_pass": bool(abs(reconstruction_error) <= tolerance),
    }


def analyze_gate(aggregates: dict, phase_a: dict, contract: dict, checks: dict) -> dict:
    tolerance = float(contract["factorial_reconstruction_absolute_tolerance"])
    metrics = {
        metric: factorial_metric(aggregates, phase_a, metric, tolerance)
        for metric in contract["gate_a"]["metrics"]
    }
    endpoint_tolerance = float(contract["endpoint_replication_absolute_tolerance"])
    endpoint_replication = all(
        abs(row["hta_endpoint_replication_error"]) <= endpoint_tolerance
        and abs(row["reference_endpoint_replication_error"]) <= endpoint_tolerance
        for row in metrics.values()
    )
    factorial_reconstruction = all(row["factorial_reconstruction_pass"] for row in metrics.values())
    minimum = min(row["attribution_fraction"] for row in metrics.values())
    integrity = all(checks.values()) and endpoint_replication and factorial_reconstruction
    passed = integrity and minimum >= float(contract["gate_a"]["threshold"])
    return {
        "threshold": float(contract["gate_a"]["threshold"]), "metrics": metrics,
        "minimum_attribution_fraction": float(minimum),
        "endpoint_replication_pass": endpoint_replication,
        "factorial_reconstruction_pass": factorial_reconstruction,
        "integrity_pass": integrity, "gate_a_pass": passed,
        "continuation_authorized": passed,
    }


def write_csv(path: Path, seed_summaries: dict):
    rows = []
    for arm, seeds in seed_summaries.items():
        for seed, summary in seeds.items():
            row = {"arm": arm, "seed": int(seed)}
            row.update({key: value for key, value in summary.items() if not isinstance(value, dict)})
            rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def render_report(payload: dict) -> str:
    gate = payload["gate_a"]
    delivery = gate["metrics"]["delivery_ratio"]
    efficiency = gate["metrics"]["packets_per_j"]
    lines = [
        "# HTA-MAC V2 Phase A2 causal/accounting repair",
        "",
        f"**Status:** `{payload['status']}`  ",
        f"**Gate A:** `{'PASS' if gate['gate_a_pass'] else 'FAIL'}` at the unchanged 0.80 threshold  ",
        "**Scope:** diagnostic MAC-only factorial replay; no Phase B and no training.",
        "",
        "## Causal attribution",
        "",
        "| Metric | A1 gap | Count-rule Shapley | Selection-rule Shapley | Unexplained residual | Attribution |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Delivery | {delivery['a1_observed_gap']:.9f} | {delivery['count_rule_shapley']:.9f} | {delivery['selection_rule_shapley']:.9f} | {delivery['unexplained_accounting_residual']:.3e} | {delivery['attribution_fraction']:.6f} |",
        f"| Packets/J | {efficiency['a1_observed_gap']:.9f} | {efficiency['count_rule_shapley']:.9f} | {efficiency['selection_rule_shapley']:.9f} | {efficiency['unexplained_accounting_residual']:.3e} | {efficiency['attribution_fraction']:.6f} |",
        "",
        f"Controlling attribution: `{gate['minimum_attribution_fraction']:.6f}`. Endpoint replication: `{gate['endpoint_replication_pass']}`. Factorial reconstruction: `{gate['factorial_reconstruction_pass']}`.",
        "",
        "The count contribution measures HTA's positive-marginal stopping versus filling all feasible demand up to budget. The selection contribution measures HTA Q-ranking versus corrected residual-energy ranking, including downstream CH/member depletion and lost future service along each replayed trajectory.",
        "",
        "## Factorial arm means",
        "",
        "| Arm | Delivery | Packets/J | RMST | Allocated slots | Selection-divergence steps |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = payload["aggregates"][arm]
        lines.append(
            f"| {arm} | {row['mean_delivery_ratio']:.6f} | {row['mean_packets_per_j']:.6f} | "
            f"{row['mean_restricted_survival_rounds']:.2f} | {row['mean_allocated_slots']:.2f} | "
            f"{row['mean_selection_disagreement_steps']:.2f} |"
        )
    lines.extend([
        "", "## Evidence boundary", "",
        "Seeds 12000--12019 were already opened by A1 and were reused for diagnosis only. Seeds 3900--3919 remained excluded. The failed A1 result is preserved and not replaced.",
        "",
        "No teacher construction, architecture change, imitation, C51/PPO training, or Phase B work was performed.",
    ])
    if gate["gate_a_pass"]:
        lines.extend(["", "The repaired causal audit passes Gate A, but this run stops at Phase A2. Phase B requires a separate authorized execution."])
    else:
        lines.extend(["", "Gate A still fails. Work stops here without weakening the gate."])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    contract_path = resolve(args.contract)
    output_dir = resolve(args.output_dir)
    report_path = resolve(args.report)
    contract, phase_a = load_contract(contract_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    completed = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(contract["workers"])) as pool:
        futures = [
            pool.submit(evaluate_seed, str(contract_path), int(seed), str(output_dir / "factorial_traces"))
            for seed in contract["development_seeds"]
        ]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            completed.append(future.result())
            if index % 2 == 0 or index == len(futures):
                print(f"PHASE_A2_PROGRESS={index}/{len(futures)}", flush=True)
    seed_summaries = {arm: {} for arm in ARMS}
    traces = []
    action_failures = 0
    for task in completed:
        action_failures += int(task["action_replication_failures"])
        for arm in ARMS:
            seed_summaries[arm][str(task["seed"])] = summarize_rows(task["episodes"][arm])
        traces.append({key: task[key] for key in ("seed", "trace", "trace_sha256", "decision_rows", "action_replication_failures")})
    aggregates = {arm: aggregate_seeds(seed_summaries[arm]) for arm in ARMS}
    expected = set(map(str, contract["development_seeds"]))
    checks = {
        "all_20_seed_tasks_complete": len(completed) == 20,
        "paired_seed_cohorts": all(set(seed_summaries[arm]) == expected for arm in ARMS),
        "opened_confirmation_seeds_excluded": not bool(
            set(contract["development_seeds"]) & set(contract["permanently_opened_confirmation_seeds"])
        ),
        "zero_action_factor_replication_failures": action_failures == 0,
        "zero_feasibility_violations": all(aggregates[arm]["feasible"] for arm in ARMS),
        "role_energy_reconstruction": all(
            aggregates[arm]["max_role_reconstruction_error_j"] <= 1e-12 for arm in ARMS
        ),
        "no_training": True, "no_phase_b": True,
    }
    gate = analyze_gate(aggregates, phase_a, contract, checks)
    payload = {
        "schema_version": 1,
        "status": "phase_a2_gate_pass_stop" if gate["gate_a_pass"] else "phase_a2_gate_fail_stop",
        "phase": "A2_only_no_training",
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "contract": str(contract_path), "contract_sha256": sha256(contract_path),
        "runner_sha256": sha256(Path(__file__)),
        "source_phase_a_results_sha256": sha256(contract["source_phase_a_results"]),
        "source_checkpoint_sha256": sha256(contract["source_checkpoint"]),
        "development_seeds": contract["development_seeds"], "checks": checks,
        "seed_summaries": seed_summaries, "aggregates": aggregates,
        "gate_a": gate, "factorial_traces": sorted(traces, key=lambda row: row["seed"]),
        "elapsed_seconds": time.perf_counter() - started,
        "original_phase_a_failure_preserved": True,
        "neural_training_performed": False, "phase_b_started": False,
        "claim_boundary": "causal diagnostic replay on already-opened development seeds; not confirmation evidence",
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "seed_level_factorial_metrics.csv", seed_summaries)
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
