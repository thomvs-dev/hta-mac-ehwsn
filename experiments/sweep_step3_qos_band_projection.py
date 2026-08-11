"""Parallel development sweep of a minimal two-sided QoS action projection."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.train_phase2_dynamic_curriculum as trainer
from agents.ch_depletion_risk import validate_ch_risk_config
from agents.qos_constraints_v3 import Step3QoSConstraintConfig
from experiments.diagnose_step3_delivery_feasibility import build_environments, load_agent, sha256
from experiments.sweep_step3_qos_deficit_override import qos_deficit_override


def qos_band_projection(
    base_action, q_values, caps, mask, env, *, lower_target, upper_target,
    reserve_floor, completion_fraction,
):
    action, addition = qos_deficit_override(
        base_action, q_values, caps, mask, env,
        trajectory_target=lower_target,
        reserve_floor=reserve_floor,
        completion_fraction=completion_fraction,
    )
    counts = env.step3_qos_counts
    offered = int(env.base.queue[env.members][env.base.alive[env.members]].sum())
    predicted_demand = int(counts["demand"]) + offered
    lower_required = int(math.ceil(float(lower_target) * predicted_demand))
    upper_allowed = int(math.floor(float(upper_target) * predicted_demand))
    predicted_delivery = int(counts["delivered"]) + int(action.sum())
    removable = min(
        max(0, predicted_delivery - upper_allowed),
        max(0, predicted_delivery - lower_required),
    )
    removed = 0
    service = np.asarray(env.cumulative_service, dtype=np.float64)
    while removed < removable and int(action.sum()) > 0:
        candidates = []
        for node in np.flatnonzero(np.asarray(mask, dtype=bool) & (action > 0)):
            level = int(action[node])
            q_loss = float(q_values[node, level] - q_values[node, level - 1])
            candidates.append((float(service[node]), -q_loss, -int(node), int(node)))
        if not candidates:
            break
        node = max(candidates)[-1]
        action[node] -= 1
        removed += 1
    return action, {
        "added": int(addition["added"]),
        "risk_blocked": int(addition["risk_blocked"]),
        "removed": removed,
        "upper_excess_before_removal": max(0, predicted_delivery - upper_allowed),
    }


def worker(payload, upper_target):
    threads = int(payload[7])
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, min(2, threads)))
    except RuntimeError:
        pass
    checkpoint, profile, risk_path, qos_path = map(Path, payload[:4])
    horizon = int(payload[4])
    lower_target, reserve_floor, completion_fraction = map(float, payload[5])
    agent, _ = load_agent(checkpoint)
    risk = validate_ch_risk_config(json.loads(risk_path.read_text()))
    qos = Step3QoSConstraintConfig.from_payload(json.loads(qos_path.read_text()))
    environments, _, cfg = build_environments(profile, risk, horizon)
    rows = []
    totals = {"added_slots": 0, "removed_slots": 0, "risk_blocked_slots": 0, "upper_excess": 0}
    for env in environments:
        observation, mask, _ = env.reset()
        done = False
        while not done:
            padded, padded_mask, caps = trainer.padded_state(env, observation, mask, env.base.n_nodes)
            base_action, q_values = agent.act(
                padded, padded_mask, epsilon=0.0, caps=caps,
                budget=cfg.frame_slot_budget,
                tie_break_priorities=np.arange(env.base.n_nodes),
            )
            action, audit = qos_band_projection(
                base_action, q_values, caps, padded_mask, env,
                lower_target=lower_target, upper_target=upper_target,
                reserve_floor=reserve_floor, completion_fraction=completion_fraction,
            )
            totals["added_slots"] += audit["added"]
            totals["removed_slots"] += audit["removed"]
            totals["risk_blocked_slots"] += audit["risk_blocked"]
            totals["upper_excess"] += audit["upper_excess_before_removal"]
            observation, mask, done, _ = env.step(action)
        counts = env.step3_qos_counts
        demand = max(1, int(counts["demand"]))
        delivery = int(counts["delivered"]) / demand
        stale = int(counts["stale"]) / demand
        fairness = float(counts["fairness"])
        rows.append({
            "target_rank": int(env.target_rank),
            "delivery_ratio": delivery,
            "stale_ratio": stale,
            "fairness": fairness,
            "joint_qos_pass": bool(
                delivery >= qos.minimum_delivery_ratio
                and stale <= qos.maximum_stale_drop_ratio
                and fairness >= qos.minimum_queue_fairness
            ),
            "fnd_free_steps": int(env.base.t_fnd if env.base.t_fnd is not None else horizon),
        })
    return {
        "upper_target": upper_target,
        "joint_qos_pass_count": int(sum(row["joint_qos_pass"] for row in rows)),
        "delivery_pass_count": int(sum(row["delivery_ratio"] >= qos.minimum_delivery_ratio for row in rows)),
        "fairness_pass_count": int(sum(row["fairness"] >= qos.minimum_queue_fairness for row in rows)),
        "mean_delivery_ratio": float(np.mean([row["delivery_ratio"] for row in rows])),
        "mean_fairness": float(np.mean([row["fairness"] for row in rows])),
        "mean_fnd_free_steps": float(np.mean([row["fnd_free_steps"] for row in rows])),
        "totals": totals,
        "pairs": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--environment-profile", type=Path, required=True)
    parser.add_argument("--ch-risk-config", type=Path, required=True)
    parser.add_argument("--qos-config", type=Path, required=True)
    parser.add_argument("--sweep-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resolve = lambda path: path if path.is_absolute() else ROOT / path
    checkpoint, profile, risk_path, qos_path, sweep_path, output = map(
        resolve, (args.checkpoint, args.environment_profile, args.ch_risk_config,
                  args.qos_config, args.sweep_config, args.output)
    )
    sweep = json.loads(sweep_path.read_text())
    if sweep.get("status") != "frozen_adaptive_development_sweep":
        raise RuntimeError("sweep is not frozen")
    frozen = (
        float(sweep["lower_delivery_target"]), float(sweep["reserve_floor"]),
        float(sweep["completion_fraction"]),
    )
    payload = (
        str(checkpoint), str(profile), str(risk_path), str(qos_path),
        int(sweep["horizon"]), frozen, int(sweep["workers"]),
        int(sweep["threads_per_worker"]),
    )
    started = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(sweep["workers"])) as pool:
        futures = [pool.submit(worker, payload, float(value)) for value in sweep["upper_delivery_targets"]]
        candidates = [future.result() for future in futures]
    candidates.sort(key=lambda row: row["upper_target"])
    feasible = [row for row in candidates if row["joint_qos_pass_count"] >= 18 and row["mean_fnd_free_steps"] >= 1158.75]
    selected = max(
        feasible,
        key=lambda row: (row["joint_qos_pass_count"], row["mean_fnd_free_steps"], -row["upper_target"]),
    ) if feasible else None
    result = {
        "schema_version": 1,
        "status": "qos_band_candidate_selected" if selected else "no_candidate_passed",
        "development_only": True,
        "publication_evidence": False,
        "checkpoint_sha256": sha256(checkpoint),
        "sweep_config_sha256": sha256(sweep_path),
        "candidates": candidates,
        "selected_upper_target": selected["upper_target"] if selected else None,
        "elapsed_seconds": time.perf_counter() - started,
        "confirmation_seed_2401_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "selected_upper_target": result["selected_upper_target"],
        "candidates": [
            {key: row[key] for key in ("upper_target", "joint_qos_pass_count", "mean_fnd_free_steps", "mean_delivery_ratio")}
            for row in candidates
        ],
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2), flush=True)
    return 0 if selected else 3


if __name__ == "__main__":
    raise SystemExit(main())
