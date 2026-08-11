"""Accept or reject the persisted episode-100 checkpoint without rerunning training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    rows = [json.loads(line) for line in args.episodes_jsonl.read_text().splitlines() if line.strip()]
    if int(metadata.get("episode", -1)) != 100 or len(rows) != 100:
        raise RuntimeError("bounded probe requires exactly 100 completed episodes")
    evaluation = metadata.get("evaluation", {})
    qos = evaluation.get("step3_target_qos", {})
    tail = rows[-50:]
    risk = [abs(float(row.get("weighted_terms", {}).get("ch_depletion_risk", 0.0))) for row in tail]
    all_terms = [sum(abs(float(value)) for value in row.get("weighted_terms", {}).values()) for row in tail]
    risk_fraction = sum(risk) / max(sum(all_terms), 1e-12)
    finite = all(np.isfinite(float(row.get("scaled_learning_reward", 0.0))) for row in rows)
    zero_fraction = float(np.mean([float(row.get("zero_action_fraction", 0.0)) for row in tail]))
    gates = {
        "exactly_100_episodes": True,
        "finite_learning_rows": finite,
        "no_always_sleep_collapse": zero_fraction <= 0.80,
        "greedy_target_qos": qos.get("pass") is True,
        "ch_risk_active": sum(risk) > 0.0,
        "ch_risk_non_dominating": risk_fraction < 0.20,
    }
    passed = all(gates.values())
    payload = {
        "schema_version": 1,
        "status": "bounded_training_checkpoint_pass" if passed else "bounded_training_checkpoint_fail_stop",
        "overall_pass": passed,
        "optimizer_seed": 5599,
        "episodes": 100,
        "full_training_authorized": False,
        "gates": gates,
        "greedy_target_qos": qos,
        "risk_tail_absolute_reward_fraction": risk_fraction,
        "tail_zero_action_fraction": zero_fraction,
        "claim_boundary": "one_seed_development_probe_not_model_selection_or_publication_evidence",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "gates": gates}, indent=2))
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__":
    main()
