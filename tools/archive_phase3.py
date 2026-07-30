"""Archive authoritative corrected Phase 3 evidence with SHA-256 hashes."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "outputs" / "archive" / "authoritative_phase3_20260728"
SOURCES = (
    ROOT / "PHASE3_STATUS.md",
    ROOT / "BASELINE_PROVENANCE.md",
    ROOT / "config" / "phase3.yaml",
    ROOT / "baselines" / "interface.py",
    ROOT / "baselines" / "policies.py",
    ROOT / "outputs" / "phase3" / "paired_pilot_5seed" / "raw_trials.csv",
    ROOT / "outputs" / "phase3" / "paired_pilot_5seed" / "summary.json",
    ROOT / "outputs" / "phase3" / "paired_pilot_5seed" / "static_idle_off_compatibility.csv",
    ROOT / "outputs" / "logs" / "phase3_pilot_audit.json",
    ROOT / "outputs" / "logs" / "phase1_gate.json",
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
    for index, source in enumerate(SOURCES, start=1):
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = ARCHIVE / f"{index:02d}_{source.name}"
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
        "status": "authoritative_corrected_phase3_evidence",
        "created_date": "2026-07-28",
        "superseded_run": "outputs/phase3/paired_pilot_5seed_superseded_revival_bug",
        "phase4_ready": False,
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
