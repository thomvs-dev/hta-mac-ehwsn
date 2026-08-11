"""Make Step 3 return-scale calibration safely resumable from Drive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook")
    args = parser.parse_args()
    path = Path(args.notebook)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    marker = "subprocess.run([\n    sys.executable, '-B', 'experiments/calibrate_step3_return_scale.py',"
    replacement = """reuse_scale = False
if scale.is_file():
    try:
        saved_scale = json.loads(scale.read_text())
        reuse_scale = (
            saved_scale.get('status') == 'frozen_development_scale'
            and saved_scale.get('development_seeds') == DEVELOPMENT_SEEDS
            and saved_scale.get('rollouts') == 100
            and saved_scale.get('episode_length_max', 0) <= TRAINING_HORIZON
        )
    except Exception:
        reuse_scale = False
if reuse_scale:
    print('Reusing verified return-scale calibration from Drive:', scale)
else:
    subprocess.run([
        sys.executable, '-B', 'experiments/calibrate_step3_return_scale.py',"""
    matches = 0
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if marker in source:
            source = source.replace(marker, replacement)
            # Indent the remainder of the subprocess call until its closing line.
            lines = source.splitlines(True)
            start = next(i for i, line in enumerate(lines) if "experiments/calibrate_step3_return_scale.py" in line)
            end = next(i for i in range(start, len(lines)) if lines[i].startswith("], cwd=repo, check=True)"))
            for index in range(start + 1, end + 1):
                lines[index] = "    " + lines[index]
            source = "".join(lines)
            cell["source"] = source.splitlines(True)
            matches += 1
    if matches != 1:
        raise RuntimeError(f"expected one calibration cell, found {matches}")
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
