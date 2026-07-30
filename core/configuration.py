"""Configuration loading for the dependency-light Phase 0 foundation."""

from __future__ import annotations

import json
from pathlib import Path


def load_simple_yaml(path: str | Path) -> dict:
    """Load the deliberately restricted YAML subset used by HTA-MAC configs."""
    path = Path(path)
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        content = raw_line.split("#", 1)[0].rstrip()
        if not content.strip():
            continue
        indent = len(content) - len(content.lstrip(" "))
        stripped = content.strip()
        if ":" not in stripped:
            raise ValueError(f"{path}:{line_number}: unsupported YAML line")
        key, raw_value = (part.strip() for part in stripped.split(":", 1))
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if not raw_value:
            value: object = {}
            parent[key] = value
            stack.append((indent, value))
            continue
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        parent[key] = value
    return root

