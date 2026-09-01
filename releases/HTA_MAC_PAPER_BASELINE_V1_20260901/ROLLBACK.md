# Safe rollback to HTA-MAC Paper Baseline V1

Rollback is intentionally performed into a new directory. Do not extract over
the active repository; that could destroy later work.

## 1. Verify the release payload

From PowerShell in the release directory:

```powershell
$manifest = Get-Content -LiteralPath .\ARTIFACT_MANIFEST.json -Raw | ConvertFrom-Json
foreach ($entry in $manifest.files) {
    $path = Join-Path (Get-Location) $entry.release_path
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing release file: $($entry.release_path)"
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLower()
    if ($actual -ne $entry.sha256) {
        throw "Checksum mismatch: $($entry.release_path)"
    }
}
"All frozen payload files verified."
```

## 2. Restore source to a separate directory

```powershell
$restore = Resolve-Path .. | ForEach-Object { Join-Path $_ 'HTA_MAC_PAPER_BASELINE_V1_RESTORED' }
New-Item -ItemType Directory -Force -Path $restore | Out-Null
Expand-Archive -LiteralPath .\source_snapshot.zip -DestinationPath $restore -Force
```

## 3. Restore the checkpoint and evidence

The executable checkpoint is stored at:

`artifacts/outputs/phase2/step3_primary_idle_hybrid_100ep_opt6801_cpu6/branching_c51.pt`

Copy the `artifacts` subtree into the restored source while retaining its
relative paths. The current manuscript is separately available under
`manuscript/` and should be compiled from there.

## 4. Confirm identity

Before evaluation, verify that the restored checkpoint SHA-256 is:

`31dc4bbed0b91ff326066dee24db3d550f6df4a347eaca82c728c4b77103934a`

Use the frozen 3,000-round, budget-24 contract. Do not retune on seeds
3900--3919. If a fresh run differs, preserve both results and diagnose the
environment rather than replacing the frozen evidence.

