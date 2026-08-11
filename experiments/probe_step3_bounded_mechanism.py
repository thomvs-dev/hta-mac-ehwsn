"""Reconcile static-policy risk reachability with paired-policy FND reachability."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-profile", type=Path, required=True)
    parser.add_argument("--ch-risk-config", type=Path, required=True)
    parser.add_argument("--headroom-evidence", type=Path, required=True)
    parser.add_argument("--seeds", default="2400,2401,2402,2403,2404")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resolve = lambda path: path if path.is_absolute() else ROOT / path
    profile, risk, headroom, output = map(resolve, (
        args.environment_profile, args.ch_risk_config,
        args.headroom_evidence, args.output,
    ))
    static_output = output.with_name(output.stem + ".static_equal.json")
    command = [
        sys.executable, "-B", str(ROOT / "experiments/probe_step3_mechanism.py"),
        "--environment-profile", str(profile), "--ch-risk-config", str(risk),
        "--seeds", args.seeds, "--max-steps", str(args.max_steps),
        "--output", str(static_output),
    ]
    run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if not static_output.is_file():
        raise RuntimeError(f"static mechanism probe produced no report: {run.stderr}")
    static = json.loads(static_output.read_text())
    paired = json.loads(headroom.read_text())
    seeds = [int(value) for value in args.seeds.split(",") if value]
    energy_rows = {
        int(row["seed"]): row for row in paired.get("raw_trials", [])
        if row.get("policy") == "energy_proportional"
    }
    static_core = {
        name: static.get("gates", {}).get(name) is True
        for name in (
            "all_seeds_activate_ch_risk",
            "risk_non_dominating_conservative_bound",
            "role_energy_reconstructs_exactly",
            "configured_horizon_covers_prior_fnd_region",
        )
    }
    paired_lifetime = bool(
        sorted(energy_rows) == seeds
        and all(row.get("t_fnd") is not None for row in energy_rows.values())
        and all(int(row.get("censor_round", -1)) == args.max_steps for row in energy_rows.values())
    )
    gates = {
        **static_core,
        "paired_energy_proportional_observes_fnd_all_seeds": paired_lifetime,
        "same_development_seeds": paired.get("seeds") == seeds,
        "same_horizon": int(paired.get("horizon", -1)) == args.max_steps,
    }
    passed = all(gates.values())
    payload = {
        "schema_version": 1,
        "status": "step3_mechanism_probe_pass" if passed else "step3_mechanism_probe_fail",
        "overall_pass": passed,
        "learning_performed": False,
        "gates": gates,
        "static_equal_death_observation_required": False,
        "reason": (
            "Risk activation/accounting are checked under static equal; FND reachability is checked under the paired energy-proportional policy. "
            "Death occurrence is policy-dependent in the idle-listening-disabled B16 profile."
        ),
        "static_probe_status": static.get("status"),
        "static_probe_sha256": sha256(static_output),
        "headroom_evidence_sha256": sha256(headroom),
        "energy_proportional_fnd_rounds": {str(seed): energy_rows[seed]["t_fnd"] for seed in seeds if seed in energy_rows},
        "claim_boundary": "mechanism_and_lifetime_reachability_only_not_model_performance",
        "static_stdout": run.stdout,
        "static_stderr": run.stderr,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "gates": gates}, indent=2))
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__":
    main()
