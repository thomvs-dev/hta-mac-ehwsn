"""Apply the frozen 100-episode activation-probe decision to a generated notebook."""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook")
    args = parser.parse_args()
    path = Path(args.notebook)
    text = path.read_text(encoding="utf-8")
    replacements = {
        "Bounded five-episode probe.": "Bounded 100-episode death-activation probe.",
        "experiments/train_step3_probe.py": "experiments/train_step3_activation_probe.py",
        "'--episodes', '5'": "'--episodes', '100'",
    }
    for old, new in replacements.items():
        if old not in text:
            raise RuntimeError(f"notebook marker missing: {old}")
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
