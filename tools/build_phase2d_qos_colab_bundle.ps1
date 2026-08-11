param([string]$OutputDirectory = (Join-Path $PSScriptRoot "..\colab"))

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Stage2 = (Resolve-Path (Join-Path $Repo "..")).Path
$FinalRepo = Join-Path $Stage2 "final_repo"
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$BuildRoot = Join-Path $OutputDirectory "_phase2d_qos_bundle_build"
$Expected = [IO.Path]::GetFullPath((Join-Path $Repo "colab\_phase2d_qos_bundle_build"))
if ([IO.Path]::GetFullPath($BuildRoot) -ne $Expected) {
    throw "Refusing unexpected build root: $BuildRoot"
}
if (Test-Path -LiteralPath $BuildRoot) {
    Remove-Item -LiteralPath $BuildRoot -Recurse -Force
}

$PackedStage2 = Join-Path $BuildRoot "stage2"
$PackedHTA = Join-Path $PackedStage2 "hta-mac"
$PackedFinal = Join-Path $PackedStage2 "final_repo"
New-Item -ItemType Directory -Force -Path $PackedHTA, $PackedFinal | Out-Null

foreach ($folder in @("agents", "baselines", "config", "core", "envs", "experiments", "validation")) {
    Copy-Item -LiteralPath (Join-Path $Repo $folder) -Destination $PackedHTA -Recurse -Force
}
foreach ($file in @(
    ".gitignore",
    "README.md",
    "HTA_MAC_MODEL_PERFORMANCE_RESEARCH_AND_FORWARD_PLAN_20260804.md",
    "HTA_MAC_PHASE2D_FOUNDATION_IMPLEMENTATION_20260804.md"
)) {
    if (Test-Path -LiteralPath (Join-Path $Repo $file)) {
        Copy-Item -LiteralPath (Join-Path $Repo $file) -Destination $PackedHTA -Force
    }
}

$PackedCache = Join-Path $PackedHTA "outputs\cache\phase3_schedules"
New-Item -ItemType Directory -Force -Path $PackedCache | Out-Null
foreach ($seed in 2300..2304) {
    $pattern = "seed_" + $seed + "_horizon_300_v2_*.pkl"
    $matches = @(Get-ChildItem -LiteralPath (Join-Path $Repo "outputs\cache\phase3_schedules") -File |
        Where-Object { $_.Name -like $pattern })
    if ($matches.Count -ne 1) {
        throw "Expected one schema-v2 schedule cache for seed $seed, found $($matches.Count)"
    }
    Copy-Item -LiteralPath $matches[0].FullName -Destination $PackedCache -Force
}

$SmokeSource = Join-Path $Repo "outputs\phase2\phase2d_qos_constrained_smoke_seed2299_50ep_20260804"
$SmokeDestination = Join-Path $PackedHTA "evidence\phase2d_qos_smoke"
New-Item -ItemType Directory -Force -Path $SmokeDestination | Out-Null
foreach ($file in @("summary.json", "phase2d_foundation_audit.json")) {
    Copy-Item -LiteralPath (Join-Path $SmokeSource $file) -Destination $SmokeDestination -Force
}

$ReferenceCheckpointSource = Join-Path $Repo "outputs\phase2\authoritative_dynamic_budget8_500ep\branching_c51.pt"
$ReferenceCheckpointDestination = Join-Path $PackedHTA "outputs\phase2\authoritative_dynamic_budget8_500ep"
New-Item -ItemType Directory -Force -Path $ReferenceCheckpointDestination | Out-Null
Copy-Item -LiteralPath $ReferenceCheckpointSource -Destination $ReferenceCheckpointDestination -Force

foreach ($folder in @("configs", "ehwsn", "env", "models", "src", "utils")) {
    Copy-Item -LiteralPath (Join-Path $FinalRepo $folder) -Destination $PackedFinal -Recurse -Force
}
Get-ChildItem -LiteralPath $FinalRepo -File |
    Where-Object { $_.Extension -in @(".py", ".toml", ".txt") } |
    Copy-Item -Destination $PackedFinal -Force
$PackedFinalOutputs = Join-Path $PackedFinal "outputs"
$PackedCheckpoints = Join-Path $PackedFinalOutputs "checkpoints"
New-Item -ItemType Directory -Force -Path $PackedCheckpoints | Out-Null
Copy-Item -LiteralPath (Join-Path $FinalRepo "outputs\stage1_params.mat") -Destination $PackedFinalOutputs -Force
Copy-Item -LiteralPath (Join-Path $FinalRepo "outputs\checkpoints\model_v91_throughput.pt") -Destination $PackedCheckpoints -Force

$CacheDirectories = @(Get-ChildItem -LiteralPath $PackedStage2 -Directory -Recurse |
    Where-Object { $_.Name -eq "__pycache__" } |
    Sort-Object { $_.FullName.Length } -Descending)
foreach ($directory in $CacheDirectories) {
    if (-not $directory.FullName.StartsWith($PackedStage2, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove cache outside build root: $($directory.FullName)"
    }
    Remove-Item -LiteralPath $directory.FullName -Recurse -Force
}
$files = @(Get-ChildItem -LiteralPath $PackedStage2 -Recurse -File | ForEach-Object {
    [ordered]@{
        path = $_.FullName.Substring($PackedStage2.Length + 1).Replace("\", "/")
        bytes = $_.Length
        sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
})
$manifest = [ordered]@{
    schema_version = 3
    purpose = "HTA-MAC Phase 2D QoS-constrained three-lineage development training"
    created_utc = [DateTime]::UtcNow.ToString("o")
    hta_mac_source_commit = (git -C $Repo rev-parse HEAD).Trim()
    frozen_heart_ch_commit = (git -C $FinalRepo rev-parse HEAD).Trim()
    optimizer_seeds = @(2299, 3299, 4299)
    development_seeds = @(2300, 2301, 2302, 2303, 2304)
    prohibited_held_out_seeds = @(3100, 3101, 3102, 3103, 3104)
    episodes = 500
    max_steps = 300
    architecture = "equivariant_set_branching"
    observation_schema = "phase2d_ttl_cap_v2"
    projection_budget = 12
    reward_scale_config = "hta-mac/config/phase2c_return_scale.json"
    qos_constraint_config = "hta-mac/config/phase2d_qos_constraints.json"
    files = $files
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $PackedStage2 "COLAB_PHASE2D_QOS_MANIFEST.json") -Encoding utf8

$zip = Join-Path $OutputDirectory "HTA_MAC_Phase2D_QoS_Training_Bundle_20260804.zip"
if (Test-Path -LiteralPath $zip) {
    Remove-Item -LiteralPath $zip -Force
}
Compress-Archive -LiteralPath $PackedStage2 -DestinationPath $zip -CompressionLevel Optimal
$hash = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
"$hash  $([IO.Path]::GetFileName($zip))" | Set-Content -LiteralPath "$zip.sha256" -Encoding ascii
Write-Output "BUNDLE=$zip"
Write-Output "BYTES=$((Get-Item -LiteralPath $zip).Length)"
Write-Output "SHA256=$hash"
Write-Output "FILES=$($files.Count)"
Remove-Item -LiteralPath $BuildRoot -Recurse -Force
