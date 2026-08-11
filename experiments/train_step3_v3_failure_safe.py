"""Failure-safe Step 3 v3 entrypoint with an episode-complete checkpoint alias."""

from __future__ import annotations

import json
from pathlib import Path

import experiments.train_step3_v3 as v3


_ORIGINAL_SAVE = v3.RecoveryExportAgent.save


def failure_safe_save(self, path, metadata=None):
    _ORIGINAL_SAVE(self, path, metadata)
    if (metadata or {}).get("episode") != 500 or v3._EXPORT_DIR is None:
        return
    source = Path(v3._EXPORT_DIR) / Path(path).name
    destination = Path(v3._EXPORT_DIR) / "training_complete_weights.pt"
    v3.atomic_copy(source, destination)
    payload = {
        "schema_version": 1,
        "status": "episode_500_weights_persisted_before_final_evaluation",
        "weights_retrained": False,
        "optimizer_seed": v3._OPTIMIZER_SEED,
        "run_name": v3._RUN_NAME,
        "source_checkpoint": source.name,
        "checkpoint_sha256": v3.sha256(destination),
        "exact_midrun_resume": False,
        "finalization_recoverable": True,
    }
    sidecar = destination.with_suffix(".json")
    temporary = sidecar.with_name(sidecar.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(sidecar)


def main():
    v3.RecoveryExportAgent.save = failure_safe_save
    return v3.main()


if __name__ == "__main__":
    raise SystemExit(main())
