"""Cheap deterministic Step 3 reachability audit; performs no learning."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.train_phase2_dynamic_curriculum as trainer
from agents.ch_depletion_risk import ch_depletion_risk, validate_ch_risk_config
from envs.policy_observation import PHASE2D_POLICY_SCHEMA
from envs.step3_lifetime_env import RoleSeparatedScheduledMACEnv


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-profile", type=Path, required=True)
    parser.add_argument("--ch-risk-config", type=Path, required=True)
    parser.add_argument("--seeds", default="2400,2401,2402,2403,2404")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resolve = lambda p: p if p.is_absolute() else ROOT / p
    profile, risk_path, output = map(resolve, (args.environment_profile, args.ch_risk_config, args.output))
    risk_config = validate_ch_risk_config(json.loads(risk_path.read_text()))
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if set(seeds) & set(range(3100, 3105)):
        raise ValueError("registered held-out seeds are prohibited")
    trainer.ScheduledIntraClusterMACEnv = RoleSeparatedScheduledMACEnv
    trials = []
    for seed in seeds:
        envs, manifest, config = trainer.build_curriculum(
            [seed], args.max_steps, observation_schema=PHASE2D_POLICY_SCHEMA,
            environment_profile=profile,
        )
        base = envs[0].base
        base.reset(seed=seed, frozen_snapshot=envs[0].bundle)
        nonzero_risk = 0
        global_deaths = 0
        risk_abs = 0.0
        delivered_weighted_abs = 0.0
        reconstruction_error = 0.0
        for _ in range(args.max_steps):
            state = base._state()
            action = base.static_equal_action()
            heads, cluster_of, alive = base.cluster_heads.copy(), base.cluster_of.copy(), base.alive.copy()
            queue = base.queue.copy()
            for cluster, ch_value in enumerate(heads):
                ch = int(ch_value)
                members = np.flatnonzero((cluster_of == cluster) & alive & (np.arange(base.n_nodes) != ch))
                intended = int(np.minimum(queue[members], action[members]).sum())
                result = ch_depletion_risk(
                    risk_config,
                    reserve_fraction=float(base.energy[ch] / base.cfg.initial_energy_j),
                    forecast_harvest_j=float(state[ch, 1]),
                    distance_to_bs_m=float(np.linalg.norm(base.positions[ch] - np.asarray(base.cfg.bs_position_m))),
                    intended_delivered_packets=intended,
                    frame_slot_budget=base.cfg.frame_slot_budget,
                )
                score = abs(float(result["raw_penalty"]))
                nonzero_risk += int(score > 0)
                risk_abs += score * float(risk_config["weight"]) / float(risk_config["scale"])
                delivered_weighted_abs += (2.0 / 1.05) * (intended / max(1, len(members)))
            alive_before = int(base.alive.sum())
            _, _, terminated, truncated, info = base.step(action)
            global_deaths += alive_before - int(base.alive.sum())
            role = info["energy_trace"]["role_energy"]
            reconstruction_error = max(
                reconstruction_error,
                float(np.max(np.abs(np.asarray(role["reconstructed_consumed"]) - np.asarray(info["energy_trace"]["consumed"])))),
            )
            if base.t_fnd is not None:
                break
            if terminated or truncated:
                break
        fraction_upper_bound = risk_abs / max(risk_abs + delivered_weighted_abs, 1e-12)
        trials.append({
            "seed": seed,
            "configured_horizon": args.max_steps,
            "steps_executed": int(base.round),
            "t_fnd": base.t_fnd,
            "global_deaths": global_deaths,
            "nonzero_ch_risk_records": nonzero_risk,
            "risk_absolute_reward_fraction_upper_bound": fraction_upper_bound,
            "maximum_role_energy_reconstruction_error_j": reconstruction_error,
            "schedule_schema_version": manifest[0]["schedule_schema_version"],
        })
    cap = float(risk_config["max_allowed_absolute_reward_fraction"])
    gates = {
        "all_seeds_observe_fnd": all(item["t_fnd"] is not None for item in trials),
        "all_seeds_observe_death": all(item["global_deaths"] > 0 for item in trials),
        "all_seeds_activate_ch_risk": all(item["nonzero_ch_risk_records"] > 0 for item in trials),
        "risk_non_dominating_conservative_bound": all(item["risk_absolute_reward_fraction_upper_bound"] <= cap for item in trials),
        "role_energy_reconstructs_exactly": all(item["maximum_role_energy_reconstruction_error_j"] <= 1e-15 for item in trials),
        "configured_horizon_covers_prior_fnd_region": args.max_steps >= 1200,
    }
    passed = all(gates.values())
    payload = {
        "schema_version": 1,
        "status": "step3_mechanism_probe_pass" if passed else "step3_mechanism_probe_fail",
        "learning_performed": False,
        "policy": "static_equal_deterministic_reachability_only",
        "claim_boundary": "signal_and_accounting_reachability_not_model_performance",
        "environment_profile_sha256": sha256(profile),
        "ch_risk_config_sha256": sha256(risk_path),
        "frozen_risk_fraction_cap": cap,
        "gates": gates,
        "trials": trials,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(output)}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
