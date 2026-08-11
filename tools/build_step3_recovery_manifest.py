"""Inventory recovered Step 3 Drive files without inferring success from filenames."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_metadata(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata", {})
        return metadata.get("optimizer_seed"), metadata.get("episode"), metadata.get("episodes_completed")
    except Exception as exc:
        return None, None, f"unreadable:{type(exc).__name__}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.directory.resolve()
    files = []
    observed_seeds = set()
    for path in sorted(item for item in source.rglob("*") if item.is_file() and item.resolve() != args.output.resolve()):
        entry = {"path": path.relative_to(source).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)}
        if path.suffix == ".pt":
            seed, episode, completed = checkpoint_metadata(path)
            entry.update(embedded_optimizer_seed=seed, embedded_episode=episode, embedded_episodes_completed=completed)
            if seed is not None:
                observed_seeds.add(int(seed))
        match = re.search(r"seed(\d+)", path.name)
        if match:
            entry["filename_seed_hint"] = int(match.group(1))
        files.append(entry)
    lineages = {
        "6499": "diagnostic_complete_not_v3_candidate" if 6499 in observed_seeds else "not_verified",
        "7499": "trained_to_episode_500_finalization_incomplete_no_checkpoint",
        "5499": "missing_not_failed",
    }
    payload = {
        "schema_version": 1,
        "status": "recovered_step3_v1_evidence_frozen",
        "source": str(source),
        "selection_eligible": False,
        "lineages": lineages,
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"FILES={len(files)}")
    print(f"OUTPUT={args.output.resolve()}")


if __name__ == "__main__":
    main()
