$ErrorActionPreference = 'Stop'
$releaseRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$manifestPath = Join-Path $releaseRoot 'ARTIFACT_MANIFEST.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json

foreach ($entry in $manifest.files) {
    $path = Join-Path $releaseRoot $entry.release_path
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing release file: $($entry.release_path)"
    }
    $file = Get-Item -LiteralPath $path
    if ($file.Length -ne $entry.bytes) {
        throw "Size mismatch: $($entry.release_path)"
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLower()
    if ($actual -ne $entry.sha256) {
        throw "Checksum mismatch: $($entry.release_path)"
    }
}

Write-Host "Verified $($manifest.files.Count) frozen files for $($manifest.release_id)."
