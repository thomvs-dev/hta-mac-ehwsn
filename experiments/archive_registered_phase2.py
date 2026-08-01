"""Freeze all preregistered Phase 2 runs into a hash-and-size manifest."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (8, 12, 16, 20, 24)
SEEDS = (2299, 3299, 4299)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def expected_runs():
    for budget in BUDGETS:
        for seed in SEEDS:
            yield "shared_branching", budget, seed, f"registered_shared_b{budget}_seed{seed}"
    for seed in SEEDS:
        yield "independent_dqns", 12, seed, f"registered_independent_b12_seed{seed}"


def main() -> int:
    registry_path = ROOT / "outputs" / "phase2" / "registered_sweep_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if not registry.get("complete") or registry.get("gate_pass_runs") != 18:
        raise RuntimeError("registered sweep registry is not complete with 18 passes")

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase2_completion_commit": git_head(),
        "preregistration": "PHASE4_PREREGISTRATION.md",
        "preregistration_sha256": sha256_file(ROOT / "PHASE4_PREREGISTRATION.md"),
        "registry": {
            "path": str(registry_path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": registry_path.stat().st_size,
            "sha256": sha256_file(registry_path),
        },
        "expected_runs": 18,
        "runs": [],
    }
    for architecture, budget, seed, name in expected_runs():
        run_dir = ROOT / "outputs" / "phase2" / name
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        checks = {
            "phase2_curriculum_gate_pass": summary.get("phase2_curriculum_gate_pass") is True,
            "episodes_completed_500": summary.get("episodes_completed") == 500,
            "architecture_matches": summary.get("agent_config", {}).get("architecture") == architecture,
            "budget_matches": summary.get("projection_budget") == budget,
            "optimizer_seed_matches": summary.get("optimizer_seed") == seed,
            "no_collapse": summary.get("always_sleep_collapse") is False,
            "no_nonfinite": summary.get("nonfinite_detected") is False,
            "convergence_pass": summary.get("convergence", {}).get("pass") is True,
            "policy_stability_pass": summary.get("policy_stability", {}).get("pass") is True,
            "trajectory_q_differentiated": summary.get("trajectory_q_check", {}).get("differentiated") is True,
        }
        if not all(checks.values()):
            raise RuntimeError(f"admission check failed for {name}: {checks}")
        files = []
        for path in sorted(run_dir.iterdir()):
            if not path.is_file():
                continue
            files.append(
                {
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        manifest["runs"].append(
            {
                "run_name": name,
                "architecture": architecture,
                "budget": budget,
                "optimizer_seed": seed,
                "training_git_hash": summary.get("git_hash"),
                "finalization_git_hash": summary.get("finalization_git_hash", summary.get("git_hash")),
                "checks": checks,
                "files": files,
            }
        )
        print(f"ARCHIVED {name} files={len(files)}", flush=True)

    manifest["run_count"] = len(manifest["runs"])
    manifest["file_count"] = sum(len(run["files"]) for run in manifest["runs"])
    output = ROOT / "outputs" / "phase2" / "registered_sweep_artifact_manifest.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"REGISTERED_RUNS_ARCHIVED={manifest['run_count']}")
    print(f"REGISTERED_FILES_HASHED={manifest['file_count']}")
    print(f"MANIFEST_SHA256={sha256_file(output)}")
    print(f"manifest={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())