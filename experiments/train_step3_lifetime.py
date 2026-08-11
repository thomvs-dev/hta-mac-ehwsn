"""Same-runtime Step 3 entry point with CH-risk reward and frozen CH schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.train_phase2_dynamic_curriculum as trainer
from agents.ch_depletion_risk import validate_ch_risk_config
from core.runtime_fingerprint import validate_runtime_contract
from envs.step3_lifetime_env import (
    RoleSeparatedScheduledMACEnv,
    Step3DynamicClusterTrainingEnv,
    configure_step3_risk,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


class Step3RewardModel:
    def __init__(self, scales, weights):
        scales["ch_depletion_risk"] = float(_RISK_CONFIG["scale"])
        weights["ch_depletion_risk"] = float(_RISK_CONFIG["weight"])
        self.scales = scales
        self.weights = weights

    def evaluate(self, raw_terms):
        weighted = {}
        for name in trainer.TERM_ORDER:
            scale = max(float(self.scales[name]), 1e-12)
            weighted[name] = float(self.weights[name]) * float(raw_terms[name]) / scale
        return float(sum(weighted.values())), weighted


_RISK_CONFIG = None


def main() -> int:
    global _RISK_CONFIG
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ch-risk-config", type=Path, required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    step3_args, remaining = parser.parse_known_args()
    risk_path = resolve(step3_args.ch_risk_config)
    contract_path = resolve(step3_args.runtime_contract)
    preflight_path = resolve(step3_args.preflight_report)
    _RISK_CONFIG = validate_ch_risk_config(json.loads(risk_path.read_text()))
    contract = json.loads(contract_path.read_text())
    runtime_validation = validate_runtime_contract(contract)
    if not runtime_validation["pass"]:
        raise RuntimeError(f"probe/training runtime mismatch: {runtime_validation}")
    preflight = json.loads(preflight_path.read_text())
    if preflight.get("status") != "step3_pretraining_claim_preflight_pass":
        raise RuntimeError("Step 3 preflight has not passed")
    if preflight.get("runtime_fingerprint_sha256") != contract["fingerprint_sha256"]:
        raise RuntimeError("preflight/runtime contract fingerprint mismatch")
    if preflight.get("ch_risk_config_sha256") != sha256(risk_path):
        raise RuntimeError("preflight/CH-risk config mismatch")

    trainer.ScheduledIntraClusterMACEnv = RoleSeparatedScheduledMACEnv
    trainer.DynamicClusterTrainingEnv = Step3DynamicClusterTrainingEnv
    trainer.TERM_ORDER = tuple(trainer.TERM_ORDER) + ("ch_depletion_risk",)
    trainer.RewardModel = Step3RewardModel
    configure_step3_risk(_RISK_CONFIG)
    sys.argv = [sys.argv[0], *remaining]
    run_name = remaining[remaining.index("--run-name") + 1]
    result = trainer.main()

    run_dir = ROOT / "outputs" / "phase2" / run_name
    summary_path = run_dir / "summary.json"
    checkpoint_path = run_dir / "branching_c51.pt"
    if summary_path.is_file() and checkpoint_path.is_file():
        summary = json.loads(summary_path.read_text())
        summary["step3_lifetime"] = {
            "status": "development_mechanism_candidate",
            "ch_schedule_modified": False,
            "learned_intervention": "mac_allocation_only",
            "ch_risk_config": str(risk_path),
            "ch_risk_config_sha256": sha256(risk_path),
            "ch_risk_config_payload": _RISK_CONFIG,
            "role_separated_energy_accounting": True,
            "runtime_contract": str(contract_path),
            "runtime_contract_sha256": sha256(contract_path),
            "runtime_validation": runtime_validation,
            "preflight_report": str(preflight_path),
            "preflight_report_sha256": sha256(preflight_path),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint["metadata"] = summary
        torch.save(checkpoint, checkpoint_path)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
