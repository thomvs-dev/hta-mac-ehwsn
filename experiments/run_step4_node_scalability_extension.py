"""Evaluate frozen HTA-MAC from 50 to 300 nodes in paired increments of 50."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
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
from agents.branching_dqn import BranchingAgentConfig, BranchingDQNAgent
from agents.qos_constraints_v3 import Step3QoSConstraintConfig
from envs.step3_policy_observation import STEP3_CH_CONTEXT_SCHEMA
from experiments.diagnose_step3_delivery_feasibility import load_agent
from experiments.distill_step3_qos_shield import set_cpu_contract
import experiments.evaluate_step4_publication_evidence as publication
from experiments.evaluate_step4_publication_evidence import OnlinePrimalDualQoS
from experiments.paper_aligned_environment import _spatial_embedding
from experiments.run_step4_final_confirmation import holm_adjust, paired, seed_summary

_ORIGINAL_INDEPENDENT_SCHEDULE_BUNDLE = publication.independent_schedule_bundle


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(value: str | Path) -> str:
    return hashlib.sha256(resolve(value).read_bytes()).hexdigest()


def load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("status") != "frozen_postconfirmation_node_scalability_extension":
        raise RuntimeError("unexpected scalability-extension contract status")
    if list(map(int, contract["evaluation_seeds"])) != list(range(3900, 3920)):
        raise RuntimeError("scalability cohort must reuse the opened confirmation seeds")
    expected_nodes = [50, 100, 150, 200, 250, 300]
    if [int(row["nodes"]) for row in contract["scenarios"]] != expected_nodes:
        raise RuntimeError("node sizes must be 50 through 300 in increments of 50")
    if int(contract["workers"]) * int(contract["threads_per_worker"]) > 16:
        raise RuntimeError("CPU worker contract exceeds logical CPU budget")
    for field in ("source_confirmation", "source_checkpoint", "qos_config", "risk_config"):
        if sha256(contract[field]) != contract[f"{field}_sha256"]:
            raise RuntimeError(f"artifact checksum mismatch: {field}")
    if contract["decision_rules"] != {
        "no_training_or_retuning": True,
        "no_model_selection": True,
        "all_sizes_reported": True,
        "failed_or_adverse_scaling_results_reported": True,
    }:
        raise RuntimeError("scalability decision rules changed")
    expansion = contract["capacity_expansion"]
    if expansion["learned_weights_changed"] or expansion["optimizer_or_training_used"]:
        raise RuntimeError("capacity expansion must not change learned weights")
    return contract


def load_capacity_agent(checkpoint_path: Path, *, nodes: int, budget: int):
    if nodes <= 100:
        agent, checkpoint = load_agent(checkpoint_path)
        return agent, checkpoint, False
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_payload = dict(checkpoint["config"])
    if int(config_payload["max_branches"]) != 100:
        raise RuntimeError("source checkpoint does not have the frozen 100-branch capacity")
    config_payload["max_branches"] = int(nodes)
    agent = BranchingDQNAgent(BranchingAgentConfig(**config_payload), device="cpu")
    for destination, field in (
        (agent.online, "online_state_dict"), (agent.target, "target_state_dict")
    ):
        state = {key: value.clone() for key, value in checkpoint[field].items()}
        state["normalized_budget"] = torch.tensor(float(budget) / float(nodes))
        destination.load_state_dict(state, strict=True)
    agent.online.eval(); agent.target.eval()
    if sum(parameter.numel() for parameter in agent.online.parameters()) != 116033:
        raise RuntimeError("capacity expansion changed the learned parameter count")
    return agent, checkpoint, True


def scalability_schedule_bundle(
    *, seed: int, nodes: int, horizon: int, field_size: float,
    bs_position: tuple[float, float], solar_states: int, thermal_states: int,
) -> dict:
    """Extend the frozen balanced schedule to 300 nodes without retraining."""
    if nodes <= 100:
        return _ORIGINAL_INDEPENDENT_SCHEDULE_BUNDLE(
            seed=seed, nodes=nodes, horizon=horizon, field_size=field_size,
            bs_position=bs_position, solar_states=solar_states,
            thermal_states=thermal_states,
        )
    if nodes > 300:
        raise ValueError("scalability extension supports at most 300 nodes")
    rng = np.random.default_rng(int(seed) + 91_003)
    positions = rng.uniform(0.0, float(field_size), size=(nodes, 2))
    embedding = _spatial_embedding(positions, field_size, bs_position)
    solar = rng.integers(0, solar_states, size=nodes, dtype=np.int64)
    thermal = rng.integers(0, thermal_states, size=nodes, dtype=np.int64)
    ch_count = max(1, int(round(nodes * 0.05)))
    rotation_period = nodes // math.gcd(nodes, ch_count)
    permutation = rng.permutation(nodes)
    frames = []
    for round_index in range(horizon):
        if round_index % rotation_period == 0:
            permutation = rng.permutation(nodes)
        start = (round_index % rotation_period) * ch_count
        indices = np.arange(start, start + ch_count) % nodes
        heads = np.sort(permutation[indices]).astype(np.int64)
        frames.append({
            # These arrays are exogenous constants. The environment copies them
            # when installing a frame, so sharing them here removes O(NH) duplicate
            # storage without changing any installed simulator value.
            "positions": positions,
            "cluster_heads": heads,
            "solar_states": solar,
            "thermal_states": thermal,
            "stgcn_embedding": embedding,
        })
    return {
        **frames[0],
        "schedule": frames,
        "schedule_metadata": {
            "schedule_schema_version": "step4_scalability_gcd_balanced_rotation_v1",
            "coverage_rounds": horizon,
            "complete": True,
            "generator": "independent_gcd_balanced_rotation",
            "num_nodes": nodes,
            "cluster_heads_per_round": ch_count,
            "requested_cluster_head_fraction": 0.05,
            "realized_cluster_head_fraction": ch_count / nodes,
            "exact_balance_period_rounds": rotation_period,
        },
    }


def capped_energy_proportional_action(env, mask, caps, budget, exponent):
    """Energy-proportional baseline with the shared queue/cap feasibility contract."""
    mask = np.asarray(mask, dtype=bool)
    caps = np.asarray(caps, dtype=np.int64)
    action = np.zeros(env.base.n_nodes, dtype=np.int64)
    scores = np.power(
        np.clip(env.base.energy / env.base.cfg.initial_energy_j, 0.0, None),
        float(exponent),
    )
    for _ in range(int(budget)):
        eligible = mask & (action < caps)
        if not np.any(eligible):
            break
        candidates = np.flatnonzero(eligible)
        # Stable node-index tie breaking matches the deterministic evaluation rule.
        winner = int(candidates[np.argmax(scores[candidates])])
        action[winner] += 1
    return action


def evaluate_task(contract_path: str, scenario: dict, policy: str, seed: int, smoke: bool):
    contract = load_contract(resolve(contract_path))
    set_cpu_contract(int(contract["threads_per_worker"]), int(seed) + len(policy))
    runtime = dict(contract)
    runtime["development_seeds"] = [int(seed)]
    runtime["observation_schema"] = STEP3_CH_CONTEXT_SCHEMA
    if smoke:
        runtime["horizon"] = 40
    # Redirect only this post-confirmation extension; the original evaluator and
    # its evidence artifact remain unchanged.
    publication.independent_schedule_bundle = scalability_schedule_bundle
    environments = publication.build_transfer_environments(runtime, scenario, trace=None)
    qos = Step3QoSConstraintConfig.from_payload(
        json.loads(resolve(contract["qos_config"]).read_text(encoding="utf-8"))
    )
    agent = None
    expanded = False
    if policy == "hta_mac":
        agent, _, expanded = load_capacity_agent(
            resolve(contract["source_checkpoint"]),
            nodes=int(scenario["nodes"]), budget=int(contract["budget"]),
        )
    rows = []
    for env in environments:
        observation, mask, _ = env.reset()
        dual = OnlinePrimalDualQoS(qos, contract["primal_dual"], int(contract["budget"]))
        done = False
        consumed = 0.0
        allocated = 0
        feasibility_violations = 0
        while not done:
            state, active, caps = trainer.padded_state(env, observation, mask, env.base.n_nodes)
            if policy == "hta_mac":
                action, _ = agent.act(
                    state, active, epsilon=0.0, caps=caps,
                    budget=int(contract["budget"]),
                    tie_break_priorities=np.arange(env.base.n_nodes),
                )
            elif policy == "energy_proportional":
                action = capped_energy_proportional_action(
                    env, active, caps, int(contract["budget"]),
                    float(contract["energy_proportional_score_exponent"]),
                )
            elif policy == "online_primal_dual":
                action = dual.action(env, active, caps)
            else:
                raise ValueError(policy)
            if int(action.sum()) > int(contract["budget"]) or np.any(action > caps):
                feasibility_violations += 1
            allocated += int(action.sum())
            observation, mask, done, info = env.step(action)
            consumed += float(np.asarray(info["energy_trace"]["consumed"]).sum())
        counts = env.step3_qos_counts
        demand = max(1, int(counts["demand"]))
        event = env.base.t_fnd is not None
        rows.append({
            "policy": policy, "seed": int(seed), "target_rank": int(env.target_rank),
            "nodes": int(scenario["nodes"]), "capacity_expanded": bool(expanded),
            "normalized_budget": float(agent.online.normalized_budget) if agent is not None else None,
            "delivery_ratio": int(counts["delivered"]) / demand,
            "stale_ratio": int(counts["stale"]) / demand,
            "fairness": float(counts["episode_service_fairness"]),
            "joint_qos_pass": bool(
                int(counts["delivered"]) / demand >= qos.minimum_delivery_ratio
                and int(counts["stale"]) / demand <= qos.maximum_stale_drop_ratio
                and float(counts["episode_service_fairness"]) >= qos.minimum_queue_fairness
            ),
            "fnd_event_observed": bool(event),
            "restricted_survival_rounds": int(
                env.base.t_fnd if event else runtime["horizon"]
            ),
            "global_packets": int(env.base.total_packets),
            "network_energy_j": consumed,
            "packets_per_j": int(env.base.total_packets) / max(consumed, 1e-12),
            "allocated_slots": allocated,
            "feasibility_violations": int(feasibility_violations),
        })
    return scenario["id"], policy, int(seed), rows


def analyze(contract: dict, raw: dict) -> dict:
    rng = np.random.default_rng(int(contract["inference"]["bootstrap_seed"]))
    results = {}
    delivery_p = {}
    for scenario in contract["scenarios"]:
        identifier = scenario["id"]
        results[identifier] = {"nodes": int(scenario["nodes"]), "policies": {}, "comparisons": {}}
        for policy in contract["policies"]:
            summaries = {
                str(seed): seed_summary(raw[identifier][policy][str(seed)])
                for seed in contract["evaluation_seeds"]
            }
            results[identifier]["policies"][policy] = {
                "seed_summaries": summaries,
                "mean": {
                    key: float(np.mean([row[key] for row in summaries.values()]))
                    for key in next(iter(summaries.values())) if key != "rank_units"
                },
            }
        hta = results[identifier]["policies"]["hta_mac"]["seed_summaries"]
        for baseline in ("energy_proportional", "online_primal_dual"):
            base = results[identifier]["policies"][baseline]["seed_summaries"]
            comparison = {}
            for metric, sign in (
                ("delivery_ratio", 1.0), ("restricted_survival_rounds", 1.0),
                ("packets_per_j", 1.0), ("stale_ratio", -1.0), ("fairness", 1.0),
            ):
                values = np.asarray([
                    sign * (hta[str(seed)][metric] - base[str(seed)][metric])
                    for seed in contract["evaluation_seeds"]
                ])
                item = paired(
                    values, alternative="greater", rng=rng,
                    resamples=int(contract["inference"]["bootstrap_resamples"]),
                )
                item["reported_difference"] = "baseline_minus_hta" if sign < 0 else "hta_minus_baseline"
                comparison[metric] = item
            relative = np.asarray([
                (hta[str(seed)]["packets_per_j"] - base[str(seed)]["packets_per_j"])
                / max(base[str(seed)]["packets_per_j"], 1e-12)
                for seed in contract["evaluation_seeds"]
            ])
            comparison["packets_per_j_relative"] = paired(
                relative, alternative="greater", rng=rng,
                resamples=int(contract["inference"]["bootstrap_resamples"]),
            )
            results[identifier]["comparisons"][baseline] = comparison
        delivery_p[identifier] = results[identifier]["comparisons"]["energy_proportional"]["delivery_ratio"]["wilcoxon_two_sided_p"]
    adjusted = holm_adjust(delivery_p)
    for identifier, value in adjusted.items():
        results[identifier]["comparisons"]["energy_proportional"]["delivery_ratio"]["holm_adjusted_two_sided_p"] = value
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    contract_path = resolve(args.contract)
    output_path = resolve(args.output)
    contract = load_contract(contract_path)
    seeds = [int(contract["smoke_seed"])] if args.smoke else list(map(int, contract["evaluation_seeds"]))
    scenarios = [contract["scenarios"][-1]] if args.smoke else contract["scenarios"]
    started = time.perf_counter()
    raw = {row["id"]: {policy: {} for policy in contract["policies"]} for row in scenarios}
    task_specs = [
        (scenario, policy, seed)
        for scenario in scenarios
        for policy in contract["policies"]
        for seed in seeds
    ]
    total_tasks = len(task_specs)
    completed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(contract["workers"])) as pool:
        specifications = iter(task_specs)
        pending = {}
        for _ in range(min(int(contract["workers"]), total_tasks)):
            scenario, policy, seed = next(specifications)
            future = pool.submit(
                evaluate_task, str(contract_path), scenario, policy, seed, bool(args.smoke)
            )
            pending[future] = (scenario["id"], policy, seed)
        while pending:
            done, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                pending.pop(future)
                scenario, policy, seed, rows = future.result()
                raw[scenario][policy][str(seed)] = rows
                completed += 1
                if completed % max(1, total_tasks // 20) == 0 or completed == total_tasks:
                    print(f"SCALABILITY_PROGRESS={completed}/{total_tasks}", flush=True)
                try:
                    next_scenario, next_policy, next_seed = next(specifications)
                except StopIteration:
                    continue
                next_future = pool.submit(
                    evaluate_task, str(contract_path), next_scenario, next_policy,
                    next_seed, bool(args.smoke)
                )
                pending[next_future] = (next_scenario["id"], next_policy, next_seed)
    checks = {
        "all_tasks_complete": all(
            len(raw[row["id"]][policy]) == len(seeds)
            for row in scenarios for policy in contract["policies"]
        ),
        "paired_seed_cohorts": all(
            set(raw[row["id"]][policy]) == set(map(str, seeds))
            for row in scenarios for policy in contract["policies"]
        ),
        "zero_feasibility_violations": all(
            trial["feasibility_violations"] == 0
            for row in scenarios for policy in contract["policies"]
            for seed_rows in raw[row["id"]][policy].values() for trial in seed_rows
        ),
        "no_training_or_retuning": True,
    }
    results = {} if args.smoke else analyze(contract, raw)
    payload = {
        "schema_version": 1,
        "status": (
            "node_scalability_smoke_pass" if args.smoke and all(checks.values())
            else "node_scalability_complete" if not args.smoke and all(checks.values())
            else "node_scalability_incomplete"
        ),
        "smoke": bool(args.smoke),
        "contract": str(contract_path), "contract_sha256": sha256(contract_path),
        "runner_sha256": sha256(Path(__file__)),
        "source_checkpoint_sha256": sha256(contract["source_checkpoint"]),
        "evaluation_seeds": [] if args.smoke else seeds,
        "scenarios": scenarios, "checks": checks, "raw": raw, "results": results,
        "elapsed_seconds": time.perf_counter() - started,
        "training_or_retuning_performed": False,
        "claim_boundary": contract["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"], "checks": checks,
        "elapsed_seconds": payload["elapsed_seconds"], "output": str(output_path),
    }, indent=2), flush=True)
    return 0 if payload["status"] in {"node_scalability_smoke_pass", "node_scalability_complete"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
