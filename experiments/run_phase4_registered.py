"""Run the preregistered HTA-MAC Phase 4 paired evaluation.

Registered mode is deliberately rigid: seeds 4000..4029, horizon 3000, six
comparison policies, five shared-branching budgets with three trained
replicates each, and the budget-12 independent-DQN architecture ablation.
Development smoke mode is the only way to use other seeds or horizons.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines import (
    EnergyProportionalPolicy,
    FFSSAdaptedPolicy,
    HarvestProportionalPolicy,
    HTAMACPolicy,
    RandomBudgetedPolicy,
    S2A2MACAdaptedPolicy,
    StaticEqualPolicy,
)
from core.paired_statistics import paired_wilcoxon_effect
from experiments.run_phase3_pilot import (
    METRICS,
    build_assets,
    file_sha256,
    finite_row,
    git_hash,
    run_one,
    schedule_bundle,
    set_seeds,
)

REGISTERED_SEEDS = tuple(range(4000, 4030))
REGISTERED_HORIZON = 3000
BUDGETS = (8, 12, 16, 20, 24)
TRAINING_SEEDS = (2299, 3299, 4299)
BASELINE_FACTORIES = (
    ("static_equal", StaticEqualPolicy),
    ("energy_proportional", lambda: EnergyProportionalPolicy(score_exponent=2.0)),
    ("harvest_proportional", lambda: HarvestProportionalPolicy(score_exponent=2.0)),
    ("s2a2mac_adapted", lambda: S2A2MACAdaptedPolicy(energy_weight=0.25)),
    ("ffss_adapted", lambda: FFSSAdaptedPolicy(margin_weight=1.0, queue_weight=0.0)),
    ("random_budgeted_diagnostic", RandomBudgetedPolicy),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(map(str, REGISTERED_SEEDS)))
    parser.add_argument("--horizon", type=int, default=REGISTERED_HORIZON)
    parser.add_argument("--run-name", default="registered_30seed_schema_v2")
    parser.add_argument("--development-smoke", action="store_true")
    parser.add_argument("--max-tasks", type=int)
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def checkpoint_specs():
    manifest_path = (
        ROOT / "outputs" / "phase2" / "registered_sweep_artifact_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    specs = []
    for run in manifest["runs"]:
        checkpoint = next(
            item for item in run["files"] if Path(item["path"]).name == "branching_c51.pt"
        )
        path = ROOT / checkpoint["path"]
        actual = file_sha256(path)
        if actual != checkpoint["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        architecture = run["architecture"]
        budget = int(run["budget"])
        training_seed = int(run["optimizer_seed"])
        if architecture == "shared_branching":
            arm = f"hta_shared_b{budget}"
        elif architecture == "independent_dqns":
            arm = "hta_independent_b12"
        else:
            raise RuntimeError(f"unregistered architecture: {architecture}")
        specs.append(
            {
                "policy_arm": arm,
                "architecture": architecture,
                "budget": budget,
                "training_seed": training_seed,
                "checkpoint": path,
                "checkpoint_sha256": actual,
            }
        )
    expected = {
        ("shared_branching", budget, seed)
        for budget in BUDGETS
        for seed in TRAINING_SEEDS
    } | {("independent_dqns", 12, seed) for seed in TRAINING_SEEDS}
    actual = {
        (item["architecture"], item["budget"], item["training_seed"])
        for item in specs
    }
    if actual != expected or len(specs) != 18:
        raise RuntimeError("Phase 2 manifest does not contain the exact 18 registered models")
    return sorted(
        specs,
        key=lambda item: (
            item["architecture"] != "shared_branching",
            item["budget"],
            item["training_seed"],
        ),
    )


def lean_policy(spec):
    policy = HTAMACPolicy(
        spec["checkpoint"],
        device="cpu",
        allocation_budget=spec["budget"],
    )
    # Evaluation uses only the online network. Releasing training-only objects
    # keeps all 18 registered models resident without excessive memory.
    policy.agent.target = None
    policy.agent.optimizer = None
    policy.agent.replay = None
    return policy


def task_key(row):
    return (
        int(row["seed"]),
        str(row["policy_arm"]),
        "" if row.get("training_seed") in (None, "") else int(row["training_seed"]),
    )


def read_existing(raw_csv):
    if not raw_csv.is_file():
        return []
    rows = []
    with raw_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed = dict(row)
            for key in (
                "seed",
                "training_seed",
                "budget",
                "t_fnd",
                "t_hnd",
                "censor_round",
                "rounds",
                "throughput",
                "packets_generated",
                "dropped_stale_packets",
                "dropped_death_packets",
                "dropped_overflow_packets",
                "alive_at_end",
            ):
                if parsed.get(key) not in (None, ""):
                    parsed[key] = int(parsed[key])
                elif key in ("training_seed", "budget", "t_fnd", "t_hnd"):
                    parsed[key] = None
            for key in METRICS:
                if key in parsed and parsed[key] not in (None, "") and key not in (
                    "t_fnd",
                    "t_hnd",
                    "rounds",
                    "throughput",
                    "packets_generated",
                    "dropped_stale_packets",
                    "dropped_death_packets",
                    "dropped_overflow_packets",
                ):
                    parsed[key] = float(parsed[key])
            for key in (
                "literature_baseline",
                "idle_enabled",
                "t_fnd_event_observed",
                "t_hnd_event_observed",
                "right_censored",
                "schedule_exhausted",
            ):
                if key in parsed:
                    parsed[key] = str(parsed[key]).lower() == "true"
            rows.append(parsed)
    return rows


def write_rows(raw_csv, rows):
    if not rows:
        return
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with raw_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def annotate(row, *, arm, architecture, budget=None, training_seed=None, checkpoint_sha=None):
    row["policy_arm"] = arm
    row["architecture"] = architecture
    row["budget"] = budget
    row["training_seed"] = training_seed
    row["checkpoint_sha256"] = checkpoint_sha
    row["schedule_schema_version"] = 2
    return row


def expected_keys(seeds, specs):
    keys = set()
    for seed in seeds:
        for name, _ in BASELINE_FACTORIES:
            keys.add((seed, name, ""))
        for spec in specs:
            keys.add((seed, spec["policy_arm"], spec["training_seed"]))
    return keys


def restricted_value(row, endpoint, tau):
    observed = bool(row[f"{endpoint}_event_observed"])
    value = row[endpoint] if observed else row["censor_round"]
    return float(min(float(value), float(tau)))


def median_iqr(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
    }


def holm_adjust(payloads):
    order = sorted(range(len(payloads)), key=lambda i: payloads[i]["p_value_raw"])
    running = 0.0
    total = len(payloads)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (total - rank) * payloads[index]["p_value_raw"])
        running = max(running, adjusted)
        payloads[index]["p_value_holm"] = running
        payloads[index]["reject_holm_0_05"] = running < 0.05


def aggregate_seed_rows(rows, tau):
    grouped = {}
    for row in rows:
        arm = row["policy_arm"]
        grouped.setdefault((arm, int(row["seed"])), []).append(row)
    aggregated = []
    for (arm, seed), subset in sorted(grouped.items()):
        architecture = subset[0]["architecture"]
        expected_replicates = 3 if architecture in (
            "shared_branching",
            "independent_dqns",
        ) else 1
        if len(subset) != expected_replicates:
            raise RuntimeError(
                f"arm={arm} seed={seed} has {len(subset)} rather than "
                f"{expected_replicates} replicates"
            )
        row = {
            "policy_arm": arm,
            "seed": seed,
            "architecture": architecture,
            "budget": subset[0].get("budget"),
            "replicates": len(subset),
            "restricted_fnd_free_time": float(
                np.mean([restricted_value(item, "t_fnd", tau) for item in subset])
            ),
            "restricted_hnd_free_time": float(
                np.mean([restricted_value(item, "t_hnd", tau) for item in subset])
            ),
            "fnd_events": int(sum(bool(item["t_fnd_event_observed"]) for item in subset)),
            "hnd_events": int(sum(bool(item["t_hnd_event_observed"]) for item in subset)),
        }
        for metric in METRICS:
            if metric in ("t_fnd", "t_hnd"):
                continue
            values = [float(item[metric]) for item in subset]
            row[metric] = float(np.mean(values))
        aggregated.append(row)
    return aggregated


def analyze(rows, seeds):
    tau = int(min(int(row["censor_round"]) for row in rows))
    aggregated = aggregate_seed_rows(rows, tau)
    indexed = {
        (row["policy_arm"], int(row["seed"])): row for row in aggregated
    }
    arms = sorted({row["policy_arm"] for row in aggregated})
    summary = {}
    for arm in arms:
        subset = [row for row in aggregated if row["policy_arm"] == arm]
        summary[arm] = {
            "restricted_fnd_free_time": median_iqr(
                [row["restricted_fnd_free_time"] for row in subset]
            ),
            "delivery_ratio": median_iqr([row["delivery_ratio"] for row in subset]),
            "restricted_hnd_free_time": median_iqr(
                [row["restricted_hnd_free_time"] for row in subset]
            ),
            "fnd_events_across_raw_replicates": int(sum(row["fnd_events"] for row in subset)),
            "hnd_events_across_raw_replicates": int(sum(row["hnd_events"] for row in subset)),
        }

    comparisons = []
    for budget in BUDGETS:
        arm = f"hta_shared_b{budget}"
        for endpoint in ("restricted_fnd_free_time", "delivery_ratio"):
            for comparator in ("static_equal", "s2a2mac_adapted"):
                first = np.asarray(
                    [indexed[(arm, seed)][endpoint] for seed in seeds],
                    dtype=np.float64,
                )
                second = np.asarray(
                    [indexed[(comparator, seed)][endpoint] for seed in seeds],
                    dtype=np.float64,
                )
                effect = paired_wilcoxon_effect(first, second)
                effect.update(
                    {
                        "hta_arm": arm,
                        "budget": budget,
                        "endpoint": endpoint,
                        "comparator": comparator,
                        "p_value_raw": effect.pop("p_value_two_sided"),
                        "difference_orientation": "hta_minus_comparator",
                        "preferred_effect_sign": "positive",
                        "hta_summary": median_iqr(first),
                        "comparator_summary": median_iqr(second),
                    }
                )
                comparisons.append(effect)
    holm_adjust(comparisons)

    frontier = []
    candidates = []
    for budget in BUDGETS:
        arm = f"hta_shared_b{budget}"
        candidates.append(
            (
                arm,
                summary[arm]["restricted_fnd_free_time"]["median"],
                summary[arm]["delivery_ratio"]["median"],
            )
        )
    random_delivery = summary["random_budgeted_diagnostic"]["delivery_ratio"]["median"]
    for arm, fnd, delivery in candidates:
        dominated = any(
            other != arm
            and other_fnd >= fnd
            and other_delivery >= delivery
            and (other_fnd > fnd or other_delivery > delivery)
            for other, other_fnd, other_delivery in candidates
        )
        frontier.append(
            {
                "policy_arm": arm,
                "median_restricted_fnd_free_time": fnd,
                "median_delivery_ratio": delivery,
                "pareto_dominated": dominated,
                "delivery_not_below_random_median": delivery >= random_delivery,
                "highlight_eligible": (not dominated) and delivery >= random_delivery,
            }
        )
    return {
        "common_restriction_round_tau": tau,
        "seed_level_aggregation": "arithmetic mean across 3 trained replicates",
        "summary_median_iqr": summary,
        "confirmatory_family_size": len(comparisons),
        "confirmatory_comparisons": comparisons,
        "pareto_budget_decision": frontier,
        "aggregated_seed_rows": aggregated,
    }


def main():
    args = parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        raise ValueError("at least one seed is required")
    if not args.development_smoke:
        if seeds != REGISTERED_SEEDS or args.horizon != REGISTERED_HORIZON:
            raise ValueError(
                "registered mode requires exactly seeds 4000..4029 and horizon 3000"
            )
        if args.max_tasks is not None:
            raise ValueError("--max-tasks is development-smoke only")
    set_seeds(seeds[0])
    specs = checkpoint_specs()
    run_dir = ROOT / "outputs" / "phase4" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = run_dir / "raw_trials.csv"
    failures_path = run_dir / "failures.jsonl"
    rows = read_existing(raw_csv)
    completed = {task_key(row) for row in rows}
    expected = expected_keys(seeds, specs)
    unknown = completed - expected
    if unknown:
        raise RuntimeError(f"raw CSV contains unregistered rows: {sorted(unknown)[:3]}")

    frozen_policy, solar, thermal, radio, config, manifest = build_assets(args.horizon)
    checkpoint_sha = manifest["checkpoint"]["sha256"]
    policies = [(spec, lean_policy(spec)) for spec in specs]
    tasks_run = 0
    schedule_metadata = {}

    def persist(row):
        nonlocal tasks_run
        rows.append(row)
        completed.add(task_key(row))
        write_rows(raw_csv, rows)
        tasks_run += 1
        print(
            f"PHASE4_PROGRESS={len(completed)}/{len(expected)} "
            f"SEED={row['seed']} ARM={row['policy_arm']} "
            f"TRAIN_SEED={row.get('training_seed')} FND={row['t_fnd']} "
            f"DELIVERY={row['delivery_ratio']:.6f}",
            flush=True,
        )

    stop = False
    for seed in seeds:
        if stop:
            break
        bundle, metadata = schedule_bundle(
            frozen_policy, seed, args.horizon, checkpoint_sha
        )
        schedule_metadata[str(seed)] = metadata
        for name, factory in BASELINE_FACTORIES:
            key = (seed, name, "")
            if key in completed:
                continue
            try:
                row = run_one(factory(), seed, bundle, solar, thermal, radio, config, True)
                persist(annotate(row, arm=name, architecture="baseline"))
            except Exception as exc:
                failure = {
                    "utc": utc_now(),
                    "seed": seed,
                    "policy_arm": name,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                with failures_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(failure) + "\n")
                print(f"PHASE4_ERROR seed={seed} arm={name} error={exc!r}", flush=True)
            if args.max_tasks is not None and tasks_run >= args.max_tasks:
                stop = True
                break
        if stop:
            break
        for spec, policy in policies:
            key = (seed, spec["policy_arm"], spec["training_seed"])
            if key in completed:
                continue
            try:
                row = run_one(policy, seed, bundle, solar, thermal, radio, config, True)
                persist(
                    annotate(
                        row,
                        arm=spec["policy_arm"],
                        architecture=spec["architecture"],
                        budget=spec["budget"],
                        training_seed=spec["training_seed"],
                        checkpoint_sha=spec["checkpoint_sha256"],
                    )
                )
            except Exception as exc:
                failure = {
                    "utc": utc_now(),
                    "seed": seed,
                    "policy_arm": spec["policy_arm"],
                    "training_seed": spec["training_seed"],
                    "checkpoint_sha256": spec["checkpoint_sha256"],
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                with failures_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(failure) + "\n")
                print(
                    f"PHASE4_ERROR seed={seed} arm={spec['policy_arm']} "
                    f"train_seed={spec['training_seed']} error={exc!r}",
                    flush=True,
                )
            if args.max_tasks is not None and tasks_run >= args.max_tasks:
                stop = True
                break

    del policies
    gc.collect()
    all_finite = all(finite_row(row) for row in rows)
    complete = completed == expected
    report = {
        "status": (
            "registered_complete"
            if complete and all_finite and not args.development_smoke
            else "development_smoke_complete"
            if complete and all_finite and args.development_smoke
            else "incomplete"
        ),
        "phase4_gate_pass": bool(complete and all_finite),
        "development_smoke": bool(args.development_smoke),
        "created_utc": utc_now(),
        "git_hash": git_hash(),
        "preregistration": "PHASE4_PREREGISTRATION.md",
        "preregistration_sha256": file_sha256(ROOT / "PHASE4_PREREGISTRATION.md"),
        "phase2_artifact_manifest_sha256": file_sha256(
            ROOT / "outputs" / "phase2" / "registered_sweep_artifact_manifest.json"
        ),
        "seeds": list(seeds),
        "horizon": int(args.horizon),
        "expected_raw_runs": len(expected),
        "completed_raw_runs": len(completed),
        "all_metrics_finite": all_finite,
        "failure_attempts_archived": (
            sum(1 for _ in failures_path.open(encoding="utf-8"))
            if failures_path.is_file()
            else 0
        ),
        "checkpoint_count": len(specs),
        "schedule_metadata": schedule_metadata,
        "analysis": analyze(rows, seeds) if complete and all_finite else None,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if complete and all_finite:
        aggregated = report["analysis"].pop("aggregated_seed_rows")
        write_rows(run_dir / "seed_level_aggregates.csv", aggregated)
        summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PHASE4_RAW_RUNS={len(completed)}/{len(expected)}")
    print(f"PHASE4_ALL_METRICS_FINITE={all_finite}")
    print(f"PHASE4_GATE_PASS={report['phase4_gate_pass']}")
    print(f"report={summary_path}")
    return 0 if report["phase4_gate_pass"] else (0 if args.development_smoke else 2)


if __name__ == "__main__":
    raise SystemExit(main())
