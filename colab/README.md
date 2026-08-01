# HTA-MAC Google Colab training bundle

Files:

- `HTA_MAC_Phase2_Colab.ipynb`: upload/open this notebook in Google Colab.
- `HTA_MAC_Colab_Training_Bundle_20260801.zip`: frozen source, HEART-CH checkpoint/HMM, and schema-v2 development schedule caches.
- `HTA_MAC_Colab_Training_Bundle_20260801.zip.sha256`: SHA-256 checksum.

## Fast start

1. Upload the ZIP to `MyDrive/HTA_MAC_Colab_Training_Bundle_20260801.zip`.
2. Open the notebook in Colab.
3. Select **Runtime > Change runtime type > GPU** (A100 or L4 preferred).
4. Run every cell in order.
5. Leave `RUN_SELECTION = "0-17"` for the full registered sweep.
6. If the runtime disconnects, run the notebook again. Completed gate-passing runs are restored from `MyDrive/HTA_MAC_Phase2_Registered` and skipped.

The notebook stops immediately if a scientific training gate fails. It does not substitute a new seed or silently continue.

## Sharding

If one Colab session cannot finish all 18 models, use separate sessions with non-overlapping selections such as:

- `RUN_SELECTION = "0-5"`
- `RUN_SELECTION = "6-11"`
- `RUN_SELECTION = "12-17"`

All sessions must use the same Google Drive output directory. A final rerun with `RUN_SELECTION = "0-17"` restores all completed models, confirms 18/18 gates, and creates the artifact manifest.
