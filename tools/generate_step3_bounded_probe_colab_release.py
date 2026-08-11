"""Generate the bounded notebook with the policy-aware mechanism preflight."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    subprocess.run([
        sys.executable, "-B", str(ROOT / "tools/generate_step3_bounded_probe_colab.py"),
        "--bundle-sha256", args.bundle_sha256, "--output", str(args.output),
    ], check=True)
    notebook = json.loads(args.output.read_text(encoding="utf-8"))
    replacements = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        before = source
        source = source.replace(
            "experiments/probe_step3_mechanism.py",
            "experiments/probe_step3_bounded_mechanism.py",
        )
        source = source.replace(
            "'--ch-risk-config',str(risk),'--seeds'",
            "'--ch-risk-config',str(risk),'--headroom-evidence',str(stage2/'headroom_evidence/step3_mac_headroom_energy_ranked_20260810.json'),'--seeds'",
        )
        if source != before:
            replacements += 1
        cell["source"] = source.splitlines(True)
        compile(source, f"release_cell_{index}", "exec")
    if replacements != 1:
        raise RuntimeError(f"expected one bounded-mechanism replacement, got {replacements}")
    args.output.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"RELEASE_NOTEBOOK_PASS={args.output}")


if __name__ == "__main__":
    main()
