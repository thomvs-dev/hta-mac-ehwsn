"""Close Phase 0 against the user-authorized corrected empirical foundation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from core.configuration import load_simple_yaml
from core.hmm import load_solar_hmm, load_thermal_auxiliary
from core.reproducibility import git_commit, sha256_file


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parent / "final_repo"
ACCEPTANCE = ROOT / "config" / "phase0_acceptance.yaml"
MANIFEST = ROOT / "core" / "frozen_assets.yaml"


def main() -> int:
    acceptance = load_simple_yaml(ACCEPTANCE)
    manifest = load_simple_yaml(MANIFEST)
    failures: list[str] = []

    if acceptance["decision"]["status"] != "authorized_corrected_foundation":
        failures.append("corrected foundation is not authorized")
    if git_commit(UPSTREAM) != manifest["upstream"]["git_commit"]:
        failures.append("upstream git commit mismatch")

    checkpoint = UPSTREAM / manifest["checkpoint"]["path"]
    if sha256_file(checkpoint) != manifest["checkpoint"]["sha256"]:
        failures.append("checkpoint hash mismatch")
    solar_path = UPSTREAM / manifest["solar_hmm"]["path"]
    if sha256_file(solar_path) != manifest["solar_hmm"]["sha256"]:
        failures.append("solar HMM hash mismatch")

    thermal_path = ROOT / acceptance["thermal"]["artifact"]
    if sha256_file(thermal_path) != acceptance["thermal"]["sha256"]:
        failures.append("thermal auxiliary hash mismatch")
    if acceptance["thermal"]["trained"] is not False:
        failures.append("thermal auxiliary must remain labelled trained=false")

    solar = load_solar_hmm(solar_path)
    thermal = load_thermal_auxiliary(thermal_path)
    evidence_path = ROOT / acceptance["baseline"]["evidence"]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    rows = evidence.get("trials", [])
    expected_trials = acceptance["baseline"]["trials"]
    expected_seeds = list(
        range(
            acceptance["baseline"]["seed_start"],
            acceptance["baseline"]["seed_start"] + expected_trials,
        )
    )
    if len(rows) != expected_trials:
        failures.append(f"raw trial count is {len(rows)}, expected {expected_trials}")
    if [row.get("seed") for row in rows] != expected_seeds:
        failures.append("raw trial seed sequence mismatch")
    if evidence["reproducibility"]["upstream_git_commit"] != git_commit(UPSTREAM):
        failures.append("raw evidence upstream commit mismatch")
    if (
        evidence["reproducibility"]["checkpoint_sha256"]
        != manifest["checkpoint"]["sha256"]
    ):
        failures.append("raw evidence checkpoint hash mismatch")

    t_fnd = np.asarray([row["t_fnd"] for row in rows], dtype=np.float64)
    if not np.all(np.isfinite(t_fnd)):
        failures.append("non-finite T_FND value in raw evidence")
    mean = float(t_fnd.mean())
    std = float(t_fnd.std(ddof=0))
    if not np.isclose(mean, acceptance["baseline"]["t_fnd_mean"], atol=1e-12):
        failures.append(f"T_FND mean mismatch: {mean}")
    if not np.isclose(
        std, acceptance["baseline"]["t_fnd_population_std"], atol=1e-12
    ):
        failures.append(f"T_FND standard deviation mismatch: {std}")

    q1, median, q3 = np.percentile(t_fnd, [25, 50, 75])
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": 0,
        "foundation": "corrected_empirical",
        "gate_pass": not failures,
        "failures": failures,
        "baseline": {
            "trials": len(rows),
            "seed_start": expected_seeds[0],
            "seed_end": expected_seeds[-1],
            "t_fnd_mean": mean,
            "t_fnd_population_std": std,
            "t_fnd_median": float(median),
            "t_fnd_iqr": float(q3 - q1),
        },
        "artifacts": {
            "upstream_commit": git_commit(UPSTREAM),
            "checkpoint_sha256": sha256_file(checkpoint),
            "solar_hmm_sha256": sha256_file(solar_path),
            "solar_provenance": solar.provenance,
            "thermal_sha256": sha256_file(thermal_path),
            "thermal_provenance": thermal.provenance,
            "thermal_trained": False,
        },
        "retired_claim": acceptance["retired_claim"],
    }
    output = ROOT / "outputs" / "logs" / "phase0_corrected_gate.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"T_FND={mean:.1f}+/-{std:.2f}")
    print(f"T_FND_MEDIAN_IQR={median:.1f}+/-{q3-q1:.2f}")
    print(f"report={output}")
    print(f"PHASE_0_CORRECTED_GATE={'PASS' if not failures else 'FAIL'}")
    for failure in failures:
        print(f"FAIL: {failure}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())

