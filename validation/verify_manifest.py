"""Fail-fast integrity verification for locked HTA-MAC artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.configuration import load_simple_yaml
from core.reproducibility import sha256_file


def _check_file(path: Path, expected_sha256: str, label: str) -> dict:
    row = {
        "label": label,
        "path": str(path),
        "exists": path.is_file(),
        "expected_sha256": expected_sha256.lower(),
    }
    if not row["exists"]:
        row.update({"actual_sha256": None, "pass": False, "reason": "missing"})
        return row
    actual = sha256_file(path)
    row.update(
        {
            "actual_sha256": actual,
            "bytes": path.stat().st_size,
            "pass": actual.lower() == expected_sha256.lower(),
            "reason": "ok" if actual.lower() == expected_sha256.lower() else "sha256_mismatch",
        }
    )
    return row


def _git_head(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def verify_locked_assets(manifest: dict) -> list[dict]:
    upstream = (ROOT / manifest["upstream"]["path"]).resolve()
    rows = []
    actual_commit = _git_head(upstream)
    expected_commit = str(manifest["upstream"]["git_commit"])
    rows.append(
        {
            "label": "upstream.git_commit",
            "path": str(upstream),
            "expected_git_commit": expected_commit,
            "actual_git_commit": actual_commit,
            "pass": actual_commit == expected_commit,
            "reason": "ok" if actual_commit == expected_commit else "git_commit_mismatch",
        }
    )
    rows.append(
        _check_file(
            upstream / manifest["checkpoint"]["path"],
            manifest["checkpoint"]["sha256"],
            "checkpoint",
        )
    )
    rows.append(
        _check_file(
            upstream / manifest["solar_hmm"]["path"],
            manifest["solar_hmm"]["sha256"],
            "solar_hmm",
        )
    )
    rows.append(
        _check_file(
            ROOT / manifest["thermal_hmm"]["auxiliary_path"],
            manifest["thermal_hmm"]["auxiliary_sha256"],
            "thermal_hmm.auxiliary",
        )
    )
    for key in (
        "historical_hta_phase2",
        "historical_hta_phase3",
        "progress_report",
        "preregistration",
        "novelty_audit",
        "instructor_action_report",
        "registered_phase2_manifest",
    ):
        entry = manifest[key]
        rows.append(_check_file(ROOT / entry["path"], entry["sha256"], key))
    return rows


def verify_registered_phase2_artifacts(manifest: dict) -> list[dict]:
    """Verify every artifact admitted to the completed registered Phase 2 sweep."""
    entry = manifest["registered_phase2_manifest"]
    path = ROOT / entry["path"]
    if not path.is_file():
        return [{
            "label": "registered_phase2.artifact_manifest",
            "path": str(path),
            "pass": False,
            "reason": "missing",
        }]
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    expected_runs = int(data["expected_runs"])
    runs = data.get("runs", [])
    rows.append({
        "label": "registered_phase2.run_count",
        "path": str(path),
        "expected_runs": expected_runs,
        "actual_runs": len(runs),
        "pass": len(runs) == expected_runs == 18,
        "reason": "ok" if len(runs) == expected_runs == 18 else "run_count_mismatch",
    })
    registry = data["registry"]
    rows.append(
        _check_file(ROOT / registry["path"], registry["sha256"], "registered_phase2.registry")
    )
    for run in runs:
        checks = run.get("checks", {})
        admitted = bool(checks) and all(value is True for value in checks.values())
        rows.append({
            "label": f"registered_phase2.admission:{run['run_name']}",
            "path": str(path),
            "pass": admitted,
            "reason": "ok" if admitted else "admission_check_failed",
        })
        for artifact in run.get("files", []):
            row = _check_file(
                ROOT / artifact["path"],
                artifact["sha256"],
                f"registered_phase2:{run['run_name']}/{Path(artifact['path']).name}",
            )
            expected_bytes = int(artifact["bytes"])
            row["expected_bytes"] = expected_bytes
            row["size_pass"] = row.get("bytes") == expected_bytes
            if not row["size_pass"]:
                row["pass"] = False
                row["reason"] = "size_mismatch"
            rows.append(row)
    return rows

def verify_archive_manifests() -> list[dict]:
    rows = []
    for manifest_path in sorted((ROOT / "outputs" / "archive").glob("*/manifest.json")):
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in data.get("files", []):
            path = manifest_path.parent / entry["archived_as"]
            row = _check_file(
                path,
                entry["sha256"],
                f"archive:{manifest_path.parent.name}/{entry['archived_as']}",
            )
            expected_bytes = int(entry["bytes"])
            row["expected_bytes"] = expected_bytes
            row["size_pass"] = row.get("bytes") == expected_bytes
            if not row["size_pass"]:
                row["pass"] = False
                row["reason"] = "size_mismatch"
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-output",
        type=Path,
        default=ROOT / "outputs" / "logs" / "manifest_verification.json",
    )
    parser.add_argument("--skip-archives", action="store_true")
    args = parser.parse_args()

    manifest_path = ROOT / "core" / "frozen_assets.yaml"
    manifest = load_simple_yaml(manifest_path)
    rows = verify_locked_assets(manifest)
    rows.extend(verify_registered_phase2_artifacts(manifest))
    if not args.skip_archives:
        rows.extend(verify_archive_manifests())
    failures = [row for row in rows if not row["pass"]]
    report = {
        "manifest": str(manifest_path.relative_to(ROOT)),
        "schedule_schema_version": manifest["schedule_contract"]["schema_version"],
        "checks": rows,
        "check_count": len(rows),
        "failure_count": len(failures),
        "pass": not failures,
    }
    output = args.json_output
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for row in rows:
        print(f"{'PASS' if row['pass'] else 'FAIL'} {row['label']}")
    print(f"MANIFEST_CHECKS={len(rows)}")
    print(f"MANIFEST_FAILURES={len(failures)}")
    print(f"ARTIFACT_MANIFEST_PASS={not failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())