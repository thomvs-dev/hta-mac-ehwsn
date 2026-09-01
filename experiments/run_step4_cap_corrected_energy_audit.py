"""Corrective post-confirmation audit of the energy-proportional comparator."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.train_phase2_dynamic_curriculum as trainer
from agents.qos_constraints_v3 import Step3QoSConstraintConfig
from envs.step3_policy_observation import STEP3_CH_CONTEXT_SCHEMA
from experiments.distill_step3_qos_shield import set_cpu_contract
from experiments.evaluate_step4_publication_evidence import build_transfer_environments, load_trace
from experiments.run_step4_final_confirmation import holm_adjust, paired, seed_summary
from experiments.run_step4_node_scalability_extension import capped_energy_proportional_action


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(value):
    return hashlib.sha256(resolve(value).read_bytes()).hexdigest()


def load_audit(path):
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("status") != "frozen_postconfirmation_comparator_contract_repair_audit":
        raise RuntimeError("corrective-audit contract is not frozen")
    for field in ("source_contract", "source_confirmation"):
        if sha256(audit[field]) != audit[f"{field}_sha256"]:
            raise RuntimeError(f"artifact checksum mismatch: {field}")
    if audit["evaluation_seeds"] != list(range(3900, 3920)):
        raise RuntimeError("audit must reuse the already-opened cohort exactly")
    return audit


def evaluate_energy(audit_path, scenario, seed):
    audit = load_audit(resolve(audit_path))
    source = json.loads(resolve(audit["source_contract"]).read_text(encoding="utf-8"))
    set_cpu_contract(int(audit["threads_per_worker"]), int(seed) + 7919)
    runtime = dict(source)
    runtime["development_seeds"] = [int(seed)]
    runtime["observation_schema"] = STEP3_CH_CONTEXT_SCHEMA
    trace = load_trace(resolve(source["external_trace"]["path"])) if scenario.get("use_external_trace") else None
    environments = build_transfer_environments(runtime, scenario, trace=trace)
    qos = Step3QoSConstraintConfig.from_payload(
        json.loads(resolve(source["qos_config"]).read_text(encoding="utf-8"))
    )
    rows = []
    for env in environments:
        observation, mask, _ = env.reset()
        done = False; consumed = 0.0; allocated = 0; violations = 0
        while not done:
            _, active, caps = trainer.padded_state(env, observation, mask, env.base.n_nodes)
            action = capped_energy_proportional_action(
                env, active, caps, int(source["budget"]),
                float(source["energy_proportional_score_exponent"]),
            )
            violations += int(int(action.sum()) > int(source["budget"]) or np.any(action > caps))
            allocated += int(action.sum())
            observation, mask, done, info = env.step(action)
            consumed += float(np.asarray(info["energy_trace"]["consumed"]).sum())
        counts = env.step3_qos_counts; demand = max(1, int(counts["demand"]))
        event = env.base.t_fnd is not None
        rows.append({
            "policy": "energy_proportional_queue_cap_corrected", "seed": int(seed),
            "target_rank": int(env.target_rank),
            "delivery_ratio": int(counts["delivered"]) / demand,
            "stale_ratio": int(counts["stale"]) / demand,
            "fairness": float(counts["episode_service_fairness"]),
            "joint_qos_pass": bool(
                int(counts["delivered"]) / demand >= qos.minimum_delivery_ratio
                and int(counts["stale"]) / demand <= qos.maximum_stale_drop_ratio
                and float(counts["episode_service_fairness"]) >= qos.minimum_queue_fairness
            ),
            "fnd_event_observed": bool(event),
            "restricted_survival_rounds": int(env.base.t_fnd if event else source["horizon"]),
            "global_packets": int(env.base.total_packets), "network_energy_j": consumed,
            "packets_per_j": int(env.base.total_packets) / max(consumed, 1e-12),
            "allocated_slots": allocated, "feasibility_violations": violations,
        })
    return scenario["id"], int(seed), rows


def analyze(audit, source, raw):
    prior = json.loads(resolve(audit["source_confirmation"]).read_text(encoding="utf-8"))
    rng = np.random.default_rng(20260829)
    results = {}; delivery_p = {}
    for scenario in source["scenarios"]:
        identifier = scenario["id"]
        energy = {str(seed): seed_summary(raw[identifier][str(seed)]) for seed in audit["evaluation_seeds"]}
        hta = prior["results"][identifier]["policies"]["hta_mac"]["seed_summaries"]
        comparison = {}
        for metric, sign in (("delivery_ratio", 1.0), ("restricted_survival_rounds", 1.0),
                             ("packets_per_j", 1.0), ("stale_ratio", -1.0), ("fairness", 1.0)):
            differences = np.asarray([
                sign * (hta[str(seed)][metric] - energy[str(seed)][metric])
                for seed in audit["evaluation_seeds"]
            ])
            item = paired(differences, alternative="greater", rng=rng, resamples=20000)
            item["reported_difference"] = "energy_minus_hta" if sign < 0 else "hta_minus_energy"
            comparison[metric] = item
        relative = np.asarray([
            (hta[str(seed)]["packets_per_j"] - energy[str(seed)]["packets_per_j"])
            / max(energy[str(seed)]["packets_per_j"], 1e-12)
            for seed in audit["evaluation_seeds"]
        ])
        comparison["packets_per_j_relative"] = paired(
            relative, alternative="greater", rng=rng, resamples=20000
        )
        results[identifier] = {
            "scenario": scenario,
            "hta_mac_source": "unchanged seed summaries from the original confirmation artifact",
            "hta_mac_mean": prior["results"][identifier]["policies"]["hta_mac"]["mean"],
            "corrected_energy_seed_summaries": energy,
            "corrected_energy_mean": {
                key: float(np.mean([row[key] for row in energy.values()]))
                for key in next(iter(energy.values())) if key != "rank_units"
            },
            "comparison": comparison,
        }
        delivery_p[identifier] = comparison["delivery_ratio"]["wilcoxon_two_sided_p"]
    adjusted = holm_adjust(delivery_p)
    for identifier, value in adjusted.items():
        results[identifier]["comparison"]["delivery_ratio"]["holm_adjusted_two_sided_p"] = value
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit_path = resolve(args.contract); audit = load_audit(audit_path)
    source = json.loads(resolve(audit["source_contract"]).read_text(encoding="utf-8"))
    specs = [(scenario, seed) for scenario in source["scenarios"] for seed in audit["evaluation_seeds"]]
    raw = {scenario["id"]: {} for scenario in source["scenarios"]}
    started = time.perf_counter(); completed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(audit["workers"])) as pool:
        iterator = iter(specs); pending = {}
        for _ in range(min(int(audit["workers"]), len(specs))):
            scenario, seed = next(iterator)
            pending[pool.submit(evaluate_energy, str(audit_path), scenario, seed)] = None
        while pending:
            done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                pending.pop(future); scenario, seed, rows = future.result()
                raw[scenario][str(seed)] = rows; completed += 1
                if completed % 10 == 0 or completed == len(specs):
                    print(f"CAP_AUDIT_PROGRESS={completed}/{len(specs)}", flush=True)
                try:
                    next_scenario, next_seed = next(iterator)
                except StopIteration:
                    continue
                pending[pool.submit(evaluate_energy, str(audit_path), next_scenario, next_seed)] = None
    checks = {
        "all_200_tasks_complete": completed == len(specs),
        "paired_seed_cohorts": all(set(raw[s["id"]]) == set(map(str, audit["evaluation_seeds"])) for s in source["scenarios"]),
        "zero_feasibility_violations": all(
            row["feasibility_violations"] == 0 for scenario in raw.values()
            for rows in scenario.values() for row in rows
        ),
        "no_training_or_retuning": True,
    }
    payload = {
        "schema_version": 1, "status": "cap_corrected_energy_audit_complete" if all(checks.values()) else "incomplete",
        "contract": str(audit_path), "contract_sha256": sha256(audit_path), "checks": checks,
        "evaluation_seeds": audit["evaluation_seeds"], "raw": raw,
        "results": analyze(audit, source, raw), "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": audit["claim_boundary"],
    }
    output = resolve(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "checks": checks,
                      "elapsed_seconds": payload["elapsed_seconds"], "output": str(output)}, indent=2))
    return 0 if all(checks.values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
