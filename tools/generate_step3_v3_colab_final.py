"""Generate, harden, and syntax-check the final Step 3 v3 notebook."""

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
        sys.executable, "-B", str(ROOT / "tools/generate_step3_v3_colab.py"),
        "--bundle-sha256", args.bundle_sha256, "--output", str(args.output),
    ], check=True)
    notebook = json.loads(args.output.read_text(encoding="utf-8"))
    old = "selected_qos_key = next(key for key in qos_candidates if key in Path(qos_selection['survivors'][0]['path']).parent.name)"
    new = "selected_qos_key = next(key for key in sorted(qos_candidates, key=len, reverse=True) if f'qos_{key}_25ep' in Path(qos_selection['survivors'][0]['path']).parent.name)"
    replacements = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if old in source:
            source = source.replace(old, new)
            cell["source"] = source.splitlines(True)
            replacements += 1
        compile(source, f"notebook_cell_{index}", "exec")
    if replacements != 1:
        raise RuntimeError(f"expected one QoS-selection hardening replacement, got {replacements}")
    args.output.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"NOTEBOOK_SYNTAX_PASS={args.output}")


if __name__ == "__main__":
    main()
