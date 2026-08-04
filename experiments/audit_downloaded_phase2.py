"""Audit a downloaded registered Phase 2 artifact without modifying it."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def median_iqr(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "q1": percentile(values, 0.25),
        "q3": percentile(values, 0.75),
        "iqr": percentile(values, 0.75) - percentile(values, 0.25),
    }


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_local_path(root: Path, recorded: str) -> Path:
    parts = Path(recorded).parts
    if len(parts) >= 4 and parts[:2] == ("outputs", "phase2"):
        return root / "runs" / Path(*parts[2:])
    if recorded == "outputs/phase2/registered_sweep_registry.json":
        return root / "registered_sweep_registry.json"
    raise ValueError(f"Unrecognized manifest path: {recorded}")


def read_episodes(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(records: list[dict], key: str) -> float:
    values = [float(row[key]) for row in records if row.get(key) is not None]
    return statistics.fmean(values) if values else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    root = args.artifact.resolve()
    registry_path = root / "registered_sweep_registry.json"
    manifest_path = root / "registered_sweep_artifact_manifest.json"
    completion_path = root / "COLAB_PHASE2_COMPLETION.json"
    registry = load_json(registry_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)

    integrity_errors: list[str] = []
    manifest_digest = sha256_file(manifest_path)
    if manifest_digest != completion.get("artifact_manifest_sha256"):
        integrity_errors.append("artifact manifest SHA-256 differs from completion record")
    if registry != completion.get("registry"):
        integrity_errors.append("registry differs from embedded completion registry")
    registry_entry = manifest["registry"]
    if registry_path.stat().st_size != registry_entry["bytes"]:
        integrity_errors.append("registry byte size differs from manifest")
    if sha256_file(registry_path) != registry_entry["sha256"]:
        integrity_errors.append("registry SHA-256 differs from manifest")

    manifest_files_checked = 0
    for run in manifest["runs"]:
        for entry in run["files"]:
            path = manifest_local_path(root, entry["path"])
            if not path.is_file():
                integrity_errors.append(f"missing: {entry['path']}")
                continue
            manifest_files_checked += 1
            if path.stat().st_size != entry["bytes"]:
                integrity_errors.append(f"byte-size mismatch: {entry['path']}")
            if sha256_file(path) != entry["sha256"]:
                integrity_errors.append(f"SHA-256 mismatch: {entry['path']}")

    rows: list[dict] = []
    all_required_checks = {
        "phase2_curriculum_gate_pass",
        "episodes_completed_500",
        "architecture_matches",
        "budget_matches",
        "optimizer_seed_matches",
        "no_collapse",
        "no_nonfinite",
        "convergence_pass",
        "policy_stability_pass",
        "trajectory_q_differentiated",
    }
    manifest_by_name = {run["run_name"]: run for run in manifest["runs"]}

    for registered in registry["runs"]:
        run_name = registered["run_name"]
        run_dir = root / "runs" / run_name
        summary = load_json(run_dir / "summary.json")
        runtime = load_json(run_dir / "colab_runtime.json")
        episodes = read_episodes(run_dir / "episodes.jsonl")
        qcheck = summary["trajectory_q_check"]
        s1 = [float(value) for value in qcheck["s1_q_values"]]
        s8 = [float(value) for value in qcheck["s8_q_values"]]
        q_scale = max(max(map(abs, s1)), max(map(abs, s8)), 1e-12)
        q_difference = float(qcheck["max_absolute_difference"])
        manifest_run = manifest_by_name[run_name]
        checks = manifest_run["checks"]
        required_checks_pass = all(checks.get(key) is True for key in all_required_checks)
        first50 = episodes[:50]
        last50 = episodes[-50:]
        greedy = summary["greedy_evaluation"]
        reward_fractions = summary["reward_balance"]["fractions"]
        stability_metrics = summary["policy_stability"]["metrics"]
        row = {
            "run_name": run_name,
            "architecture": registered["architecture"],
            "budget": int(registered["budget"]),
            "optimizer_seed": int(registered["optimizer_seed"]),
            "gate_pass": bool(summary["phase2_curriculum_gate_pass"]),
            "required_manifest_checks_pass": required_checks_pass,
            "episodes": len(episodes),
            "global_steps": int(summary["global_steps"]),
            "runtime_minutes": float(runtime["elapsed_seconds"]) / 60.0,
            "parameter_count": int(summary["online_parameter_count"]),
            "convergence_relative_change": float(summary["convergence"]["relative_change"]),
            "greedy_reward": float(greedy["mean_reward"]),
            "greedy_packets_per_step": float(greedy["mean_packets_per_step"]),
            "greedy_zero_action_fraction": float(greedy["mean_zero_action_fraction"]),
            "greedy_fnd_free_steps": float(greedy["mean_fnd_free_steps"]),
            "greedy_throughput": float(greedy["mean_throughput"]),
            "greedy_queue_fairness": float(greedy["mean_queue_fairness"]),
            "greedy_delivery_ratio": float(greedy["mean_delivery_ratio"]),
            "q_max_absolute_difference": q_difference,
            "q_relative_difference": q_difference / q_scale,
            "q_s1_argmax": max(range(len(s1)), key=s1.__getitem__),
            "q_s8_argmax": max(range(len(s8)), key=s8.__getitem__),
            "q_argmax_changed": max(range(len(s1)), key=s1.__getitem__)
            != max(range(len(s8)), key=s8.__getitem__),
            "reward_packet_fraction": float(reward_fractions["packets_delivered"]),
            "reward_idle_fraction": float(reward_fractions["idle_energy_j"]),
            "reward_death_fraction": float(reward_fractions["deaths"]),
            "reward_harvest_fraction": float(reward_fractions["high_harvest_alignment"]),
            "reward_declining_fraction": float(reward_fractions["declining_allocation"]),
            "reward_fairness_fraction": float(reward_fractions["queue_fairness"]),
            "stability_fnd_relative_span": float(
                stability_metrics["mean_fnd_free_steps"]["relative_span"]
            ),
            "stability_throughput_relative_span": float(
                stability_metrics["mean_throughput"]["relative_span"]
            ),
            "stability_fairness_relative_span": float(
                stability_metrics["mean_queue_fairness"]["relative_span"]
            ),
            "first50_reward": mean(first50, "reward"),
            "last50_reward": mean(last50, "reward"),
            "first50_packets_per_step": mean(first50, "packets_per_step"),
            "last50_packets_per_step": mean(last50, "packets_per_step"),
            "first50_zero_action_fraction": mean(first50, "zero_action_fraction"),
            "last50_zero_action_fraction": mean(last50, "zero_action_fraction"),
            "first50_stale_drops": mean(first50, "dropped_stale_packets"),
            "last50_stale_drops": mean(last50, "dropped_stale_packets"),
            "last50_mean_loss": mean(last50, "mean_loss"),
        }
        rows.append(row)

    grouped: dict[str, dict] = {}
    buckets: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row["architecture"], row["budget"])].append(row)
    group_metrics = [
        "greedy_reward",
        "greedy_packets_per_step",
        "greedy_zero_action_fraction",
        "greedy_fnd_free_steps",
        "greedy_throughput",
        "greedy_queue_fairness",
        "greedy_delivery_ratio",
        "q_relative_difference",
        "runtime_minutes",
    ]
    for (architecture, budget), group_rows in sorted(buckets.items()):
        key = f"{architecture}_budget_{budget}"
        grouped[key] = {
            "architecture": architecture,
            "budget": budget,
            "runs": len(group_rows),
            "metrics": {
                metric: median_iqr([float(row[metric]) for row in group_rows])
                for metric in group_metrics
            },
        }

    report = {
        "artifact": str(root),
        "integrity": {
            "manifest_sha256": manifest_digest,
            "manifest_files_checked": manifest_files_checked,
            "errors": integrity_errors,
            "pass": not integrity_errors,
        },
        "registry": {
            "complete": registry["complete"],
            "gate_pass_runs": registry["gate_pass_runs"],
            "expected_runs": registry["expected_runs"],
            "source_git_commit": registry["source_git_commit"],
            "gpu": registry["gpu"],
        },
        "all_required_checks_pass": all(
            row["required_manifest_checks_pass"] for row in rows
        ),
        "q_probe": {
            "argmax_changed_runs": sum(row["q_argmax_changed"] for row in rows),
            "relative_difference": median_iqr(
                [float(row["q_relative_difference"]) for row in rows]
            ),
            "absolute_difference": median_iqr(
                [float(row["q_max_absolute_difference"]) for row in rows]
            ),
        },
        "runtime": {
            "total_hours": sum(float(row["runtime_minutes"]) for row in rows) / 60.0,
            "per_run_minutes": median_iqr(
                [float(row["runtime_minutes"]) for row in rows]
            ),
        },
        "groups": grouped,
        "runs": rows,
    }

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({key: report[key] for key in report if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()
