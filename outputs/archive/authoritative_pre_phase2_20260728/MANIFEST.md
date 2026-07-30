# Authoritative pre-Phase-2 archive

This archive is the post-instructor Phase 1 boundary. Phase 2 training may use
only this configuration/evidence lineage or an explicitly superseding archive.

| File | SHA-256 |
|---|---|
| `cluster_contention_analysis.json` | `14bf0e6cb8bb36731045d0721cf40c648ec888fb49eca5cc0381bd9e312f78db` |
| `harvest_feature_validation.json` | `4b271e1398527e83efd21a77f528902a1fe08879b44bb7aa2fc2e06fcc7d6dd2` |
| `phase0_acceptance.yaml` | `3deb5727d0070839391cf4edde827eea0da73993e80f3b9c73efce90e6801b9d` |
| `phase0_corrected_gate.json` | `17c7bc42b0f78580daf31c299acce7100146da4bd2b0a1a2a0ed75585a812512` |
| `phase1.yaml` | `16bc9e0c44707ed1c8f7d9d63db6aa05c0dadb50226b1b687e49a8442e5b8c6a` |
| `phase1_gate.json` | `0b5b414bdbb6066c048ae4e39dbe8c6e4057a6f55124138ffdf0494b226bbc53` |
| `queue_capacity_calibration.json` | `6947f0d47efedd069ee86cbb88e913e762a6a37943c5dd8626e9b068d29bfa8a` |

Definitive gate:

```powershell
$env:PYTHONPATH='F:\WSN\matlab\stage2\hta-mac'
python -B validation\run_phase1_gate.py
```

Expected terminal line:

```text
AUTHORITATIVE_PHASE_1_GATE=PASS
```
