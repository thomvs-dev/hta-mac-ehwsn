"""Execute and verify the preregistered schema-v2 Phase 2 training sweep."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (8, 12, 16, 20, 24)
TRAINING_SEEDS = (2299, 3299, 4299)
DEVELOPMENT_SEEDS = "2300,2301,2302,2303,2304"


def registered_runs():
    runs = []
    for budget in BUDGETS:
        for seed in TRAINING_SEEDS:
            runs.append(
                {
                    "architecture": "shared_branching",
                    "budget": budget,
                    "optimizer_seed": seed,
                    "run_name": f"registered_shared_b{budget}_seed{seed}",
                }
            )
    for seed in TRAINING_SEEDS:
        runs.append(
            {
                "architecture": "independent_dqns",
                "budget": 12,
                "optimizer_seed": seed,
                "run_name": f"registered_independent_b12_seed{seed}",
            }
        )
    return runs


def command_for(run, episodes, max_steps, device):
    return [
        sys.executable,
        "-B",
        str(ROOT / "experiments" / "train_phase2_dynamic_curriculum.py"),
        "--episodes",
        str(episodes),
        "--max-steps",
        str(max_steps),
        "--development-seeds",
        DEVELOPMENT_SEEDS,
        "--optimizer-seed",
        str(run["optimizer_seed"]),
        "--projection-budget",
        str(run["budget"]),
        "--architecture",
        run["architecture"],
        "--run-name",
        run["run_name"],
        "--device",
        device,
    ]


def read_existing(run):
    summary_path = ROOT / "outputs" / "phase2" / run["run_name"] / "summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "optimizer_seed": run["optimizer_seed"],
        "projection_budget": run["budget"],
        "schedule_schema_version": 2,
    }
    actual = {key: summary.get(key) for key in expected}
    actual["architecture"] = summary.get("agent_config", {}).get("architecture")
    expected["architecture"] = run["architecture"]
    if actual != expected:
        raise RuntimeError(
            f"existing run metadata mismatch for {run['run_name']}: "
            f"expected={expected}, actual={actual}"
        )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-after-failure", action="store_true")
    args = parser.parse_args()
    if args.episodes != 500 or args.max_steps != 300:
        print("DEVELOPMENT_OVERRIDE=True")

    registry = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration": "PHASE4_PREREGISTRATION.md",
        "schedule_schema_version": 2,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "runs": [],
    }
    for run in registered_runs():
        command = command_for(run, args.episodes, args.max_steps, args.device)
        row = dict(run)
        row["command"] = command
        existing = read_existing(run)
        if existing is not None and existing.get("phase2_curriculum_gate_pass"):
            row["status"] = "existing_gate_pass"
            row["summary"] = str(
                Path("outputs") / "phase2" / run["run_name"] / "summary.json"
            )
            registry["runs"].append(row)
            continue
        if args.dry_run:
            row["status"] = "planned"
            registry["runs"].append(row)
            continue
        print(
            f"START architecture={run['architecture']} budget={run['budget']} "
            f"seed={run['optimizer_seed']}",
            flush=True,
        )
        completed = subprocess.run(command, cwd=ROOT, check=False)
        summary = read_existing(run)
        gate_pass = bool(
            summary is not None and summary.get("phase2_curriculum_gate_pass")
        )
        row["return_code"] = int(completed.returncode)
        row["status"] = "gate_pass" if gate_pass else "gate_fail"
        row["summary"] = str(
            Path("outputs") / "phase2" / run["run_name"] / "summary.json"
        )
        registry["runs"].append(row)
        if not gate_pass and not args.continue_after_failure:
            break

    expected = len(registered_runs())
    passed = sum(
        row["status"] in {"gate_pass", "existing_gate_pass"}
        for row in registry["runs"]
    )
    registry["expected_runs"] = expected
    registry["gate_pass_runs"] = passed
    registry["complete"] = len(registry["runs"]) == expected and passed == expected
    output = ROOT / "outputs" / "phase2" / "registered_sweep_registry.json"
    output.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f"REGISTERED_RUNS={expected}")
    print(f"GATE_PASS_RUNS={passed}")
    print(f"REGISTERED_SWEEP_COMPLETE={registry['complete']}")
    print(f"registry={output}")
    if args.dry_run:
        return 0
    return 0 if registry["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())