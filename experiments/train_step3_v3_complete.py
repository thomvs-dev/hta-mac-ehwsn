"""Complete Step 3 v3 trainer: energy efficiency plus failure-safe persistence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.train_step3_v3 as v3


class EnergyAuditedStep3V3Env(v3.Step3V3DynamicClusterTrainingEnv):
    def reset(self):
        result = super().reset()
        self.step3_energy_consumed_j = 0.0
        return result

    def step(self, action):
        result = super().step(action)
        info = result[3]
        self.step3_energy_consumed_j += float(
            np.asarray(info["energy_trace"]["consumed"]).sum()
        )
        return result


_BASE_GREEDY = v3.v3_greedy_evaluation
_BASE_SAVE = v3.RecoveryExportAgent.save


def complete_greedy(agent, environments, max_branches, reward_model):
    evaluation, first_env, first_observation = _BASE_GREEDY(
        agent, environments, max_branches, reward_model
    )
    efficiencies = []
    for row, env in zip(evaluation["clusters"], environments):
        consumed = float(env.step3_energy_consumed_j)
        efficiency = (
            float(row["global_throughput"] / consumed) if consumed > 0 else 0.0
        )
        row["energy_consumed_j"] = consumed
        row["packets_per_joule"] = efficiency
        efficiencies.append(efficiency)
    evaluation["mean_energy_consumed_j"] = float(
        np.mean([row["energy_consumed_j"] for row in evaluation["clusters"]])
    )
    evaluation["mean_packets_per_joule"] = float(np.mean(efficiencies))
    return evaluation, first_env, first_observation


def failure_safe_save(self, path, metadata=None):
    _BASE_SAVE(self, path, metadata)
    if (metadata or {}).get("episode") != 500 or v3._EXPORT_DIR is None:
        return
    source = Path(v3._EXPORT_DIR) / Path(path).name
    destination = Path(v3._EXPORT_DIR) / "training_complete_weights.pt"
    v3.atomic_copy(source, destination)
    payload = {
        "schema_version": 1,
        "status": "episode_500_weights_persisted_before_final_evaluation",
        "optimizer_seed": v3._OPTIMIZER_SEED,
        "run_name": v3._RUN_NAME,
        "checkpoint_sha256": v3.sha256(destination),
        "exact_midrun_resume": False,
        "finalization_recoverable": True,
    }
    sidecar = destination.with_suffix(".json")
    temporary = sidecar.with_name(sidecar.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(sidecar)


def main():
    v3.Step3V3DynamicClusterTrainingEnv = EnergyAuditedStep3V3Env
    v3.v3_greedy_evaluation = complete_greedy
    v3.RecoveryExportAgent.save = failure_safe_save
    return v3.main()


if __name__ == "__main__":
    raise SystemExit(main())
