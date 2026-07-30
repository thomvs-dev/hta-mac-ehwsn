"""Create a hash-manifested immutable copy of authoritative Phase 2 evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "outputs" / "archive" / "authoritative_phase2_20260728"
SOURCES = (
    ROOT / "PHASE2_STATUS.md",
    ROOT / "config" / "agent.yaml",
    ROOT / "config" / "reward_calibration.json",
    ROOT / "outputs" / "logs" / "phase2_reward_calibration.json",
    ROOT / "outputs" / "logs" / "phase2_fixed_cluster_controls.json",
    ROOT / "outputs" / "phase2" / "authoritative_500ep_seed2100" / "summary.json",
    ROOT / "outputs" / "phase2" / "authoritative_500ep_seed2100" / "episodes.jsonl",
    ROOT / "outputs" / "phase2" / "authoritative_500ep_seed2100" / "branching_c51.pt",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    if ARCHIVE.exists():
        raise RuntimeError(f"refusing to overwrite existing archive: {ARCHIVE}")
    ARCHIVE.mkdir(parents=True)
    manifest = []
    for source in SOURCES:
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = ARCHIVE / source.name
        shutil.copy2(source, destination)
        manifest.append(
            {
                "source": str(source.relative_to(ROOT)).replace("\\", "/"),
                "archived_as": destination.name,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
    payload = {
        "status": "immutable_authoritative_phase2_evidence",
        "created_date": "2026-07-28",
        "files": manifest,
    }
    (ARCHIVE / "manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"ARCHIVED_FILES={len(manifest)}")
    print(f"archive={ARCHIVE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
