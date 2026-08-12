"""Preregistered final matched-simulator evaluation on independent seed units."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.train_phase2_dynamic_curriculum as trainer
from agents.budget_projection import project_slot_budget
from agents.ch_depletion_risk import validate_ch_risk_config
from agents.qos_constraints_v3 import Step3QoSConstraintConfig
from baselines.policies import _rank_proportional
from experiments.diagnose_step3_delivery_feasibility import build_environments, load_agent
from experiments.distill_step3_qos_shield import set_cpu_contract
from experiments.sweep_step3_primary_listwise_residual import (
    SetRemovalRanker,
    apply_ranker,
    base_and_added,
    removal_count,
)
from experiments.sweep_step3_qos_band_projection import qos_band_projection


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha256(path):
    return hashlib.sha256(resolve(path).read_bytes()).hexdigest()


def load_contract(path):
    contract = json.loads(path.read_text())
    if contract.get("status") != "frozen_before_final_matched_baseline_evaluation":
        raise RuntimeError("final evaluation contract is not frozen")
    for field in ("source_checkpoint", "selected_ranker_checkpoint", "qos_config", "risk_config"):
        if sha256(contract[field]) != contract[f"{field}_sha256"]:
            raise RuntimeError(f"artifact hash mismatch: {field}")
    if set(contract["evaluation_seeds"]).intersection(contract["all_prior_seeds"]):
        raise RuntimeError("final evaluation cohort overlaps prior evidence")
    if int(contract["workers"]) * int(contract["threads_per_worker"]) > 16:
        raise RuntimeError("CPU contract exceeds 16 logical workers")
    expected_family = len(contract["inferential_comparators"]) * len(contract["statistics"]["metrics"])
    if expected_family != 15:
        raise RuntimeError("prespecified inferential family must contain 15 hypotheses")
    return contract


def load_ranker(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    ranker = SetRemovalRanker(int(payload["features"]), int(payload["hidden"]))
    ranker.load_state_dict(payload["state_dict"])
    ranker.eval()
    return ranker


def energy_proportional_action(env, mask, budget, exponent):
    action = np.zeros(env.base.n_nodes, dtype=np.int64)
    members = np.flatnonzero(np.asarray(mask, dtype=bool))
    if len(members):
        scores = np.power(
            env.base.energy[members] / env.base.cfg.initial_energy_j,
            float(exponent),
        )
        action[members] = _rank_proportional(scores, budget, env.base.cfg.n_max)
    return action


def s2a2mac_action(env, mask, budget, energy_weight):
    action = np.zeros(env.base.n_nodes, dtype=np.int64)
    if (int(env.target_cluster) + int(env.base.round)) % 2 == 0:
        return action
    members = np.flatnonzero(np.asarray(mask, dtype=bool))
    if not len(members):
        return action
    energy = np.clip(env.base.energy[members] / env.base.cfg.initial_energy_j, 0.0, 1.0)
    load = np.clip(env.base.queue[members] / env.base.cfg.queue_max_packets, 0.0, 1.0)
    score = float(energy_weight) * energy + (1.0 - float(energy_weight)) * load
    order = np.argsort(score, kind="stable")
    layers = np.ones(len(members), dtype=np.int64)
    layers[order[len(members) // 3 : (2 * len(members)) // 3]] = 2
    layers[order[(2 * len(members)) // 3 :]] = 3
    q_values = np.zeros((len(members), env.base.cfg.n_max + 1))
    for local, desired in enumerate(layers):
        q_values[local, : desired + 1] = np.arange(desired + 1)
        if desired < env.base.cfg.n_max:
            q_values[local, desired + 1 :] = desired - 1.0
    action[members] = project_slot_budget(q_values, budget, stop_at_nonpositive_gain=True)
    return action


def ffss_action(env, mask, budget, margin_weight, queue_weight):
    """Target-cluster form of the documented fixed-frame feasible-first adaptation."""
    action = np.zeros(env.base.n_nodes, dtype=np.int64)
    members = np.flatnonzero(np.asarray(mask, dtype=bool))
    if not len(members):
        return action
    ch = int(env.ch)
    forecast = np.maximum(env.base._state()[:, 1], 0.0)
    distance = np.linalg.norm(env.base.positions[members] - env.base.positions[ch], axis=1)
    required = np.asarray([env.base.radio.tx(env.base.cfg.packet_bits, float(d)) for d in distance])
    available = env.base.energy[members] + forecast[members]
    margin = available - required
    qualified = (env.base.queue[members] > 0) & (margin >= 0.0)
    normalized_margin = margin / max(float(np.max(np.abs(margin))), 1e-12)
    normalized_queue = env.base.queue[members] / env.base.cfg.queue_max_packets
    score = float(margin_weight) * normalized_margin + float(queue_weight) * normalized_queue
    priority = np.lexsort((members, -score, ~qualified))
    selected = members[priority[: min(int(budget), len(members))]]
    action[selected] = 1
    return action


def evaluate(policy, environments, agent, ranker, qos, contract):
    rows = []
    params, budget = contract["baseline_parameters"], int(contract["budget"])
    for env in environments:
        observation, mask, _ = env.reset()
        done, energy, steps, allocated = False, 0.0, 0, 0
        while not done:
            padded, padded_mask, caps = trainer.padded_state(env, observation, mask, env.base.n_nodes)
            if policy in {"learned_listwise_residual", "analytic_teacher"}:
                _, padded_mask, caps, base, q_values, added = base_and_added(
                    agent, env, observation, mask, contract
                )
                if policy == "learned_listwise_residual":
                    count = removal_count(added, env, contract["band"])
                    action = apply_ranker(ranker, env, q_values, added, caps, padded_mask, count)
                else:
                    band = contract["band"]
                    action, _ = qos_band_projection(
                        base, q_values, caps, padded_mask, env,
                        lower_target=float(band["lower_delivery_target"]),
                        upper_target=float(band["upper_delivery_target"]),
                        reserve_floor=float(band["reserve_floor"]),
                        completion_fraction=float(band["completion_fraction"]),
                    )
            elif policy == "energy_proportional_tuned":
                action = energy_proportional_action(
                    env, padded_mask, budget, params["energy_proportional_score_exponent"]
                )
            elif policy == "s2a2mac_adapted":
                action = s2a2mac_action(env, padded_mask, budget, params["s2a2mac_energy_weight"])
            elif policy == "ffss_adapted":
                action = ffss_action(
                    env, padded_mask, budget,
                    params["ffss_margin_weight"], params["ffss_queue_weight"],
                )
            else:
                raise ValueError(policy)
            allocated += int(action.sum())
            observation, mask, done, info = env.step(action)
            energy += float(np.asarray(info["energy_trace"]["consumed"]).sum())
            steps += 1
        counts = env.step3_qos_counts
        demand = max(1, int(counts["demand"]))
        delivery = int(counts["delivered"]) / demand
        stale = int(counts["stale"]) / demand
        fairness = float(counts["episode_service_fairness"])
        fnd = int(env.base.t_fnd if env.base.t_fnd is not None else contract["horizon"])
        rows.append({
            "policy": policy,
            "seed": int(env.seed),
            "target_rank": int(env.target_rank),
            "delivery_ratio": delivery,
            "stale_ratio": stale,
            "episode_service_fairness": fairness,
            "joint_qos_pass": bool(
                delivery >= qos.minimum_delivery_ratio
                and stale <= qos.maximum_stale_drop_ratio
                and fairness >= qos.minimum_queue_fairness
            ),
            "fnd_free_steps": fnd,
            "fnd_event_observed": bool(env.base.t_fnd is not None),
            "global_packets_delivered": int(env.base.total_packets),
            "network_energy_j": energy,
            "global_packets_per_j": int(env.base.total_packets) / max(energy, 1e-12),
            "allocated_slots": allocated,
            "steps": steps,
        })
    return rows


def worker(contract_path_string, policy):
    contract = load_contract(resolve(contract_path_string))
    set_cpu_contract(
        int(contract["threads_per_worker"]),
        int(contract["statistics"]["random_seed"]) + contract["policies"].index(policy),
    )
    risk = validate_ch_risk_config(json.loads(resolve(contract["risk_config"]).read_text()))
    qos = Step3QoSConstraintConfig.from_payload(json.loads(resolve(contract["qos_config"]).read_text()))
    agent, _ = load_agent(resolve(contract["source_checkpoint"]))
    ranker = load_ranker(resolve(contract["selected_ranker_checkpoint"]))
    environments, _, _ = build_environments(
        None, risk, int(contract["horizon"]), seeds=contract["evaluation_seeds"]
    )
    return policy, evaluate(policy, environments, agent, ranker, qos, contract)


def aggregate(rows):
    fields = (
        "delivery_ratio", "stale_ratio", "episode_service_fairness",
        "fnd_free_steps", "global_packets_delivered", "network_energy_j",
        "global_packets_per_j", "allocated_slots",
    )
    return {
        "rows": len(rows),
        "seeds": len({row["seed"] for row in rows}),
        "joint_qos_pass_count": int(sum(row["joint_qos_pass"] for row in rows)),
        "fnd_event_count": int(sum(row["fnd_event_observed"] for row in rows)),
        **{f"mean_{field}": float(np.mean([row[field] for row in rows])) for field in fields},
    }


def seed_means(rows, metrics):
    result = {}
    for seed in sorted({row["seed"] for row in rows}):
        selected = [row for row in rows if row["seed"] == seed]
        result[seed] = {
            **{metric: float(np.mean([row[metric] for row in selected])) for metric in metrics},
            "joint_qos_pass_rate": float(np.mean([row["joint_qos_pass"] for row in selected])),
        }
    return result


def paired_metric(primary, comparator, metric, direction, contract, rng):
    seeds = sorted(set(primary).intersection(comparator))
    raw = np.asarray([primary[s][metric] - comparator[s][metric] for s in seeds], dtype=np.float64)
    favorable = raw if direction == "higher" else -raw
    resamples = int(contract["statistics"]["paired_bootstrap_resamples"])
    alpha = 1.0 - float(contract["statistics"]["confidence_level"])
    bootstrap = raw[rng.integers(0, len(raw), size=(resamples, len(raw)))].mean(axis=1)
    nonzero = favorable[~np.isclose(favorable, 0.0)]
    if not len(nonzero):
        p_value, rank_biserial = 1.0, 0.0
    else:
        p_value = float(wilcoxon(favorable, alternative="two-sided", zero_method="wilcox").pvalue)
        ranks = rankdata(np.abs(nonzero))
        positive, negative = ranks[nonzero > 0].sum(), ranks[nonzero < 0].sum()
        rank_biserial = float((positive - negative) / max(positive + negative, 1e-12))
    sd = float(favorable.std(ddof=1))
    return {
        "metric": metric,
        "direction": direction,
        "orientation": "learned_listwise_residual_minus_comparator",
        "seed_units": len(seeds),
        "mean_raw_difference": float(raw.mean()),
        "mean_favorable_difference": float(favorable.mean()),
        "bootstrap_95_ci_raw_difference": [
            float(np.quantile(bootstrap, alpha / 2)),
            float(np.quantile(bootstrap, 1.0 - alpha / 2)),
        ],
        "wilcoxon_p_value_unadjusted": p_value,
        "wilcoxon_p_value_holm": None,
        "reject_holm_0_05": None,
        "paired_cohens_dz_favorable": float(favorable.mean() / sd) if sd > 0 else None,
        "paired_rank_biserial_favorable": rank_biserial,
        "wins_ties_losses_favorable": {
            "wins": int((favorable > 0).sum()),
            "ties": int(np.isclose(favorable, 0.0).sum()),
            "losses": int((favorable < 0).sum()),
        },
    }


def holm_adjust(records):
    ordered = sorted(records, key=lambda item: item["wilcoxon_p_value_unadjusted"])
    running, total = 0.0, len(ordered)
    for index, item in enumerate(ordered):
        running = max(running, (total - index) * item["wilcoxon_p_value_unadjusted"])
        item["wilcoxon_p_value_holm"] = float(min(1.0, running))
        item["reject_holm_0_05"] = bool(item["wilcoxon_p_value_holm"] <= 0.05)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract_path, output_path = resolve(args.contract), resolve(args.output)
    contract = load_contract(contract_path)
    started, policy_rows = time.perf_counter(), {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(contract["workers"])) as pool:
        futures = [pool.submit(worker, str(contract_path), policy) for policy in contract["policies"]]
        for future in concurrent.futures.as_completed(futures):
            policy, rows = future.result()
            policy_rows[policy] = rows
            print(f"POLICY_COMPLETE={policy} ROWS={len(rows)}", flush=True)

    metrics = contract["statistics"]["metrics"]
    means = {policy: seed_means(rows, metrics) for policy, rows in policy_rows.items()}
    rng = np.random.default_rng(int(contract["statistics"]["random_seed"]))
    inferential, family = {}, []
    for comparator in contract["inferential_comparators"]:
        records = [
            paired_metric(
                means["learned_listwise_residual"], means[comparator], metric,
                direction, contract, rng,
            )
            for metric, direction in metrics.items()
        ]
        inferential[comparator] = {record["metric"]: record for record in records}
        family.extend(records)
    holm_adjust(family)

    mechanism = {}
    for metric, direction in metrics.items():
        mechanism[metric] = paired_metric(
            means["learned_listwise_residual"], means["analytic_teacher"],
            metric, direction, contract, rng,
        )

    gates = contract["integrity_gates"]
    expected_rows = int(gates["expected_rows_per_policy"])
    expected_seeds = int(gates["expected_seed_units_per_policy"])
    aggregates = {policy: aggregate(rows) for policy, rows in policy_rows.items()}
    pair_keys = {
        policy: {(row["seed"], row["target_rank"]) for row in rows}
        for policy, rows in policy_rows.items()
    }
    checks = {
        "expected_rows": all(value["rows"] == expected_rows for value in aggregates.values()),
        "expected_seed_units": all(value["seeds"] == expected_seeds for value in aggregates.values()),
        "complete_pairing": len({frozenset(keys) for keys in pair_keys.values()}) == 1,
        "primary_joint_qos": aggregates["learned_listwise_residual"]["joint_qos_pass_count"] >= int(gates["minimum_primary_joint_qos_pairs"]),
        "all_fnd_events_observed": all(value["fnd_event_count"] == expected_rows for value in aggregates.values()),
    }
    payload = {
        "schema_version": 1,
        "status": "final_matched_baseline_evaluation_complete" if all(checks.values()) else "final_matched_baseline_integrity_failure",
        "contract_sha256": sha256(contract_path),
        "evaluator_sha256": sha256(Path(__file__)),
        "evaluation_seeds": contract["evaluation_seeds"],
        "evaluation_seeds_opened": True,
        "independent_unit": contract["independent_unit"],
        "nested_subcases_per_seed": contract["nested_subcases_per_seed"],
        "aggregates": aggregates,
        "seed_means": means,
        "paired_inference_holm_family": inferential,
        "mechanism_reference_unadjusted": mechanism,
        "rows": [row for policy in contract["policies"] for row in policy_rows[policy]],
        "checks": checks,
        "integrity_pass": all(checks.values()),
        "selection_or_retuning_performed": False,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": contract["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "checks": checks, "aggregates": aggregates}, indent=2))
    print(f"OUTPUT={output_path}")
    return 0 if payload["integrity_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
