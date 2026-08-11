"""Harden the generated notebook against ambiguous ZIP directory metadata."""

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
    old = """        target = WORK.joinpath(*path.parts)\n        if member.is_dir(): target.mkdir(parents=True, exist_ok=True)\n        else:\n            target.parent.mkdir(parents=True, exist_ok=True)\n            with archive.open(member) as src, target.open('wb') as dst: shutil.copyfileobj(src, dst)\n"""
    new = """        target = WORK.joinpath(*path.parts)\n        # Directory entries are metadata only. Skipping them avoids a\n        # platform-dependent ZIP directory/file ambiguity; file parents are\n        # created from the verified manifest paths below.\n        if member.is_dir() or member.filename.endswith(('/', '\\\\')):\n            continue\n        target.parent.mkdir(parents=True, exist_ok=True)\n        with archive.open(member) as src, target.open('wb') as dst:\n            shutil.copyfileobj(src, dst)\n"""
    matches = 0
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell["source"])
        if old in source:
            source = source.replace(old, new)
            cell["source"] = source.splitlines(True)
            matches += 1
    if matches != 1:
        raise RuntimeError(f"expected one extraction block, found {matches}")
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
