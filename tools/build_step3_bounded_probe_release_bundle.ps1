param([string]$OutputDirectory = (Join-Path $PSScriptRoot "..\colab"))

$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "build_step3_bounded_probe_bundle.ps1") -OutputDirectory $OutputDirectory
if ($LASTEXITCODE -ne 0) { throw "Base bounded bundle build failed" }
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
$zip = Join-Path $OutputDirectory "HTA_MAC_Step3_Bounded_Probe_Bundle_20260810.zip"
$hash = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
$notebook = Join-Path $OutputDirectory "HTA_MAC_Step3_Bounded_Probe_Colab_20260810.ipynb"
python -B (Join-Path $PSScriptRoot "generate_step3_bounded_probe_colab_release.py") --bundle-sha256 $hash --output $notebook
if ($LASTEXITCODE -ne 0) { throw "Release notebook generation failed" }
Write-Output "RELEASE_BUNDLE=$zip"
Write-Output "RELEASE_SHA256=$hash"
Write-Output "RELEASE_NOTEBOOK=$notebook"
