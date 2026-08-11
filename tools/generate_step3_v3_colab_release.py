"""Create the release notebook and switch all training calls to the complete v3 entrypoint."""

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
        sys.executable, "-B", str(ROOT / "tools/generate_step3_v3_colab_final.py"),
        "--bundle-sha256", args.bundle_sha256, "--output", str(args.output),
    ], check=True)
    notebook = json.loads(args.output.read_text(encoding="utf-8"))
    count = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        before = source
        source = source.replace("experiments/train_step3_v3.py", "experiments/train_step3_v3_complete.py")
        source = source.replace("experiments/train_step3_v3_failure_safe.py", "experiments/train_step3_v3_complete.py")
        count += int(source != before)
        cell["source"] = source.splitlines(True)
        compile(source, f"release_cell_{index}", "exec")
    if count != 2:
        raise RuntimeError(f"expected two complete-trainer cell replacements, got {count}")
    args.output.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"RELEASE_NOTEBOOK_SYNTAX_PASS={args.output}")


if __name__ == "__main__":
    main()
