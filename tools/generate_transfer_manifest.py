"""Create the external integrity anchor for the report-plus-bundle handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.repo.resolve()
    paths = [
        root / "colab/HTA_MAC_PaperAligned_B16_QoSRepaired_Training_Bundle_20260808.zip",
        root / "colab/HTA_MAC_PaperAligned_B16_QoSRepaired_Training_Bundle_20260808.zip.sha256",
        root / "colab/HTA_MAC_PaperAligned_B16_QoSRepaired_Training_Colab_20260808.ipynb",
        root / "HTA_MAC_QOS_REPAIRED_AGENT_EXECUTION_HANDOFF_20260808.md",
        root / "HTA_MAC_EXTERNAL_REVIEW_RESOLUTION_AND_PREFLIGHT_DECISION_20260808.md",
        root / "config/paper_aligned_hasani2025_architecture_decision_repaired.json",
        root / "outputs/audits/paper_aligned_b16_current_code_preflight_foundation_seed5299_20260808.json",
        root / "outputs/phase3/paper_aligned_b16_budget_pressure_audit_seed5299_20260808/summary.json",
        root / "outputs/phase3/paper_aligned_b16_budget_pressure_audit_seed5299_20260808/raw_trials.csv",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing transfer artifacts: {missing}")
    bundle = paths[0]
    sidecar_tokens = paths[1].read_text(encoding="ascii").strip().split()
    bundle_hash = sha256(bundle)
    if not sidecar_tokens or sidecar_tokens[0].lower() != bundle_hash:
        raise RuntimeError("bundle sidecar does not match bundle SHA-256")
    artifacts = []
    for path in paths:
        artifacts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    payload = {
        "schema_version": 1,
        "status": "frozen_external_transfer_integrity_anchor",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "handoff_mechanism": "transfer_manifest_plus_report_plus_preflight_addendum_plus_bundle_plus_sidecar_plus_notebook",
        "report_is_standalone_integrity_anchor": False,
        "reason": "embedding the bundle hash inside a report that is itself bundled creates a circular hash dependency",
        "bundle_sha256": bundle_hash,
        "architecture": "equivariant_set_branching",
        "track_role": "secondary_literature_alignment_side_study",
        "artifacts": artifacts,
    }
    output = args.output or root / "HTA_MAC_TRANSFER_MANIFEST_20260808.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(bundle_hash)


if __name__ == "__main__":
    main()
