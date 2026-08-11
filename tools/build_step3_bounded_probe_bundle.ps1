param([string]$OutputDirectory = (Join-Path $PSScriptRoot "..\colab"))

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Stage2 = (Resolve-Path (Join-Path $Repo "..")).Path
$FinalRepo = Join-Path $Stage2 "final_repo"
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$BuildRoot = Join-Path $OutputDirectory "_step3_bounded_probe_build"
$Expected = [IO.Path]::GetFullPath((Join-Path $Repo "colab\_step3_bounded_probe_build"))
if ([IO.Path]::GetFullPath($BuildRoot) -ne $Expected) { throw "Unexpected build root: $BuildRoot" }
if (Test-Path -LiteralPath $BuildRoot) { Remove-Item -LiteralPath $BuildRoot -Recurse -Force }
$PackedStage2 = Join-Path $BuildRoot "stage2"
$PackedHTA = Join-Path $PackedStage2 "hta-mac"
$PackedFinal = Join-Path $PackedStage2 "final_repo"
New-Item -ItemType Directory -Force -Path $PackedHTA,$PackedFinal | Out-Null
foreach ($folder in @("agents","baselines","config","core","envs","experiments","validation")) {
    Copy-Item -LiteralPath (Join-Path $Repo $folder) -Destination $PackedHTA -Recurse -Force
}
foreach ($folder in @("configs","ehwsn","env","models","src","utils")) {
    Copy-Item -LiteralPath (Join-Path $FinalRepo $folder) -Destination $PackedFinal -Recurse -Force
}
Get-ChildItem -LiteralPath $FinalRepo -File | Where-Object { $_.Extension -in @(".py",".toml",".txt") } | Copy-Item -Destination $PackedFinal -Force
$PackedOutputs = Join-Path $PackedFinal "outputs"
$PackedCheckpoints = Join-Path $PackedOutputs "checkpoints"
New-Item -ItemType Directory -Force -Path $PackedCheckpoints | Out-Null
Copy-Item -LiteralPath (Join-Path $FinalRepo "outputs\stage1_params.mat") -Destination $PackedOutputs -Force
Copy-Item -LiteralPath (Join-Path $FinalRepo "outputs\checkpoints\model_v91_throughput.pt") -Destination $PackedCheckpoints -Force

$Fixture = Join-Path $Repo "outputs\phase2\authoritative_dynamic_budget8_500ep"
$PackedFixture = Join-Path $PackedHTA "outputs\phase2\authoritative_dynamic_budget8_500ep"
New-Item -ItemType Directory -Force -Path $PackedFixture | Out-Null
foreach ($file in @("branching_c51.pt","summary.json")) {
    $source = Join-Path $Fixture $file
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing fixture $source" }
    Copy-Item -LiteralPath $source -Destination $PackedFixture -Force
}

$Headroom = Join-Path $PackedStage2 "headroom_evidence"
New-Item -ItemType Directory -Force -Path $Headroom | Out-Null
foreach ($file in @("step3_mac_headroom_local_20260810.json","step3_mac_headroom_energy_ranked_20260810.json","STEP3_BOUNDED_PROBE_DECISION_20260810.json")) {
    $source = Join-Path $Repo "outputs\audits\$file"
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing headroom evidence $source" }
    Copy-Item -LiteralPath $source -Destination $Headroom -Force
}

$Caches = @(Get-ChildItem -LiteralPath $PackedStage2 -Directory -Recurse | Where-Object { $_.Name -eq "__pycache__" } | Sort-Object { $_.FullName.Length } -Descending)
foreach ($cache in $Caches) {
    if (-not $cache.FullName.StartsWith($PackedStage2,[StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe cache target" }
    Remove-Item -LiteralPath $cache.FullName -Recurse -Force
}
$files = @(Get-ChildItem -LiteralPath $PackedStage2 -Recurse -File | ForEach-Object {
    [ordered]@{path=$_.FullName.Substring($PackedStage2.Length+1).Replace("\","/");bytes=$_.Length;sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
})
$Decision = Join-Path $Headroom "STEP3_BOUNDED_PROBE_DECISION_20260810.json"
$manifest=[ordered]@{
    schema_version=1
    purpose="One-seed 100-episode Step 3 decision probe"
    claim_boundary="development_probe_only_not_publication_or_model_selection_evidence"
    created_utc=[DateTime]::UtcNow.ToString("o")
    optimizer_seed=5599
    training_seeds=@(2400)
    evaluation_seeds=@(2400,2401,2402,2403,2404)
    prohibited_registered_seeds=@(3100,3101,3102,3103,3104)
    reserved_confirmation_seeds=@(3400,3401,3402,3403,3404)
    episodes=100
    horizon=1200
    observation_schema="step3_ch_context_v3"
    risk_config="hta-mac/config/step3_v3_risk_weight_5.json"
    qos_config="hta-mac/config/step3_v3_qos_ema_floor_candidate.json"
    full_training_authorized=$false
    hyperparameter_sweep=$false
    ch_schedule_modified=$false
    exact_same_runtime_required=$true
    bounded_probe_decision_sha256=(Get-FileHash -LiteralPath $Decision -Algorithm SHA256).Hash.ToLowerInvariant()
    files=$files
}
$ManifestPath=Join-Path $PackedStage2 "COLAB_STEP3_BOUNDED_PROBE_MANIFEST.json"
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ManifestPath -Encoding utf8
$zip=Join-Path $OutputDirectory "HTA_MAC_Step3_Bounded_Probe_Bundle_20260810.zip"
if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
Compress-Archive -LiteralPath $PackedStage2 -DestinationPath $zip -CompressionLevel Optimal
$hash=(Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
"$hash  $([IO.Path]::GetFileName($zip))" | Set-Content -LiteralPath "$zip.sha256" -Encoding ascii
$notebook=Join-Path $OutputDirectory "HTA_MAC_Step3_Bounded_Probe_Colab_20260810.ipynb"
python -B (Join-Path $Repo "tools\generate_step3_bounded_probe_colab.py") --bundle-sha256 $hash --output $notebook
if ($LASTEXITCODE -ne 0) { throw "Notebook generation failed" }
Write-Output "BUNDLE=$zip"
Write-Output "SHA256=$hash"
Write-Output "NOTEBOOK=$notebook"
Write-Output "FILES=$($files.Count)"
Remove-Item -LiteralPath $BuildRoot -Recurse -Force
