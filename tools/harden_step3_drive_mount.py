"""Add explicit Drive-auth handling and an opt-in local-output fallback."""

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

    settings_marker = "DOWNLOAD_RESULTS_WHEN_COMPLETE = True\n"
    settings_replacement = (
        settings_marker
        + "# Keep False for long runs. Set True only if Drive authentication cannot be repaired.\n"
        + "ALLOW_LOCAL_OUTPUT_WITHOUT_DRIVE = False\n"
    )
    mount_old = """from google.colab import drive
drive.mount('/content/drive')
bundle = Path(BUNDLE_PATH)
"""
    mount_new = """from google.colab import drive
drive_ready = Path('/content/drive/MyDrive').is_dir()
if drive_ready:
    print('Google Drive is already mounted; reusing the existing mount.')
else:
    try:
        drive.mount('/content/drive')
        drive_ready = Path('/content/drive/MyDrive').is_dir()
        if not drive_ready:
            raise RuntimeError('Drive mount returned without a usable MyDrive directory.')
    except Exception as exc:
        if not ALLOW_LOCAL_OUTPUT_WITHOUT_DRIVE:
            raise RuntimeError(
                'Google Drive authentication failed before training. Use Runtime > Disconnect '
                'and delete runtime, reload Colab, permit the Drive authorization prompt, and '
                'rerun. For a temporary local-only run, set '
                'ALLOW_LOCAL_OUTPUT_WITHOUT_DRIVE=True in the settings cell; local output is '
                'not persistent if the runtime disconnects.'
            ) from exc
        DRIVE_OUTPUT_DIR = '/content/HTA_MAC_Step3_CHRole_Lifetime_20260808'
        print('WARNING: Drive unavailable; using non-persistent local output:', DRIVE_OUTPUT_DIR)
bundle = Path(BUNDLE_PATH)
"""

    settings_count = mount_count = 0
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell["source"])
        if settings_marker in source and "ALLOW_LOCAL_OUTPUT_WITHOUT_DRIVE" not in source:
            source = source.replace(settings_marker, settings_replacement)
            settings_count += 1
        if mount_old in source:
            source = source.replace(mount_old, mount_new)
            mount_count += 1
        cell["source"] = source.splitlines(True)
    if settings_count != 1 or mount_count != 1:
        raise RuntimeError(
            f"expected one settings and mount block, found {settings_count}/{mount_count}"
        )
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
