"""Audit HERMES artifacts for strict Phase 0 compatibility.

The script is read-only with respect to HERMES. It records why each candidate
can or cannot serve as the frozen HEART-CH/thermal-HMM foundation for HTA-MAC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_record(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    snapshot = checkpoint.get("config_snapshot", {})
    evaluation = checkpoint.get("eval_results", {})
    compatible = (
        snapshot.get("TEMPORAL_WINDOW") == 10
        and snapshot.get("NUM_NODE_FEATURES") == 31
        and snapshot.get("ENVIRONMENT_KIND", "legacy") == "legacy"
        and snapshot.get("CH_SELECTION_MODE", "fixed_topk") == "fixed_topk"
    )
    return {
        "path": str(path),
        "sha256": sha256(path),
        "episode": checkpoint.get("episode"),
        "temporal_window": snapshot.get("TEMPORAL_WINDOW"),
        "node_features": snapshot.get("NUM_NODE_FEATURES"),
        "environment": snapshot.get("ENVIRONMENT_KIND", "legacy"),
        "ch_selection": snapshot.get("CH_SELECTION_MODE", "fixed_topk"),
        "embedded_mean_t_fnd": evaluation.get("mean_t_fnd"),
        "phase0_compatible": compatible,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hermes", type=Path, default=PROJECT_ROOT.parent / "hermes"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hermes = args.hermes.resolve()
    sys.path.insert(0, str(hermes / "src"))
    from hermes_wsn.config import HermesConfig

    config = HermesConfig()
    stage1_path = (
        hermes
        / "colab"
        / "_build"
        / "hermes_colab_bundle"
        / "heart_policy"
        / "outputs"
        / "stage1_params.mat"
    )
    stage1_keys = sorted(
        key for key in loadmat(stage1_path).keys() if not key.startswith("__")
    )
    checkpoints = [
        checkpoint_record(path) for path in sorted(hermes.rglob("*.pt"))
    ]
    compatible_checkpoints = [
        item for item in checkpoints if item["phase0_compatible"]
    ]

    learned_eval_path = (
        hermes / "wsn_results_quick_seed42" / "learned_policy_evaluation.json"
    )
    learned_eval = json.loads(learned_eval_path.read_text(encoding="utf-8"))
    learned_metadata = learned_eval["metadata"]
    learned_summary = learned_eval["summary"]

    incompatibilities = []
    if config.temporal_window != 10:
        incompatibilities.append(
            f"standalone HERMES temporal window is {config.temporal_window}, expected 10"
        )
    if config.radio.max_power_range_m != 50.0:
        incompatibilities.append(
            "standalone HERMES maximum power range is "
            f"{config.radio.max_power_range_m} m, expected 50 m"
        )
    if not {"thermal_A", "thermal_mu", "thermal_sigma2", "thermal_pi0"}.issubset(
        stage1_keys
    ):
        incompatibilities.append(
            "bundled Stage 1 MAT has no fitted thermal HMM parameter keys"
        )
    if not compatible_checkpoints:
        incompatibilities.append(
            "no HERMES checkpoint matches W=10, F=31, legacy fixed-Top-K HEART-CH"
        )
    if learned_metadata.get("hermes_included") is not False:
        incompatibilities.append(
            "learned-policy evaluation does not explicitly exclude HERMES"
        )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hermes_root": str(hermes),
        "phase0_compatible": not incompatibilities,
        "incompatibilities": incompatibilities,
        "standalone_config": {
            "num_nodes": config.num_nodes,
            "field_size": config.field_size,
            "bs_position": config.bs_position,
            "initial_energy_j": config.initial_energy_j,
            "ch_ratio": config.ch_ratio,
            "temporal_window": config.temporal_window,
            "max_power_range_m": config.radio.max_power_range_m,
            "mobile_fraction": config.mobile_fraction,
            "speed_range_m_per_round": config.speed_range_m_per_round,
            "thermal_persistence": config.harvest.persistence,
        },
        "bundled_stage1": {
            "path": str(stage1_path),
            "sha256": sha256(stage1_path),
            "keys": stage1_keys,
        },
        "checkpoints": checkpoints,
        "compatible_checkpoints": compatible_checkpoints,
        "learned_policy_evaluation": {
            "path": str(learned_eval_path),
            "environment": learned_metadata.get("environment"),
            "ch_selection": learned_metadata.get("ch_selection"),
            "temporal_window": learned_metadata.get("temporal_window"),
            "node_features": learned_metadata.get("node_features"),
            "hermes_included": learned_metadata.get("hermes_included"),
            "trials": learned_summary.get("trials"),
            "fnd_median": learned_summary.get("fnd", {}).get("median"),
            "fnd_iqr": learned_summary.get("fnd", {}).get("iqr"),
        },
        "decision": (
            "HERMES artifacts cannot replace the missing frozen HEART-CH "
            "checkpoint provenance or trained thermal HMM."
        ),
    }
    output = PROJECT_ROOT / "outputs" / "logs" / "hermes_artifact_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report={output}")
    print(f"PHASE0_COMPATIBLE={report['phase0_compatible']}")
    for reason in incompatibilities:
        print(f"INCOMPATIBLE: {reason}")
    return 0 if report["phase0_compatible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

