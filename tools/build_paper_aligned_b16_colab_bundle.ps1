param([string]$OutputDirectory = (Join-Path $PSScriptRoot "..\colab"))

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Stage2 = (Resolve-Path (Join-Path $Repo "..")).Path
$FinalRepo = Join-Path $Stage2 "final_repo"
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$BuildRoot = Join-Path $OutputDirectory "_paper_aligned_b16_bundle_build"
$Expected = [IO.Path]::GetFullPath((Join-Path $Repo "colab\_paper_aligned_b16_bundle_build"))
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
    "HTA_MAC_RELATED_PAPER_PERFORMANCE_COMPARISON_20260806.md",
    "HTA_MAC_PHASE2D_QOS_TRAINED_RESULTS_ANALYSIS_AND_RECOVERY_PLAN_20260806.md"
)) {
    if (Test-Path -LiteralPath (Join-Path $Repo $file)) {
        Copy-Item -LiteralPath (Join-Path $Repo $file) -Destination $PackedHTA -Force
    }
}
$AuthoritativeSource = Join-Path $Repo "outputs\phase2\authoritative_dynamic_budget8_500ep"
$PackedAuthoritative = Join-Path $PackedHTA "outputs\phase2\authoritative_dynamic_budget8_500ep"
New-Item -ItemType Directory -Force -Path $PackedAuthoritative | Out-Null
foreach ($file in @("branching_c51.pt", "summary.json")) {
    $source = Join-Path $AuthoritativeSource $file
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Required shared-contract artifact is missing: $source"
    }
    Copy-Item -LiteralPath $source -Destination $PackedAuthoritative -Force
}


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
$Profile = Join-Path $Repo "config\paper_aligned_hasani2025_b16.json"
$Constraints = Join-Path $Repo "config\paper_aligned_hasani2025_qos_constraints.json"
$manifest = [ordered]@{
    schema_version = 1
    purpose = "HTA-MAC exploratory paper-aligned terrestrial-solar B16 training"
    claim_boundary = "paper_aligned_not_a_third_party_reproduction"
    created_utc = [DateTime]::UtcNow.ToString("o")
    hta_mac_source_commit = (git -C $Repo rev-parse HEAD).Trim()
    frozen_heart_ch_commit = (git -C $FinalRepo rev-parse HEAD).Trim()
    optimizer_seeds = @(5299, 6299, 7299)
    development_seeds = @(2400, 2401, 2402, 2403, 2404)
    reserved_confirmation_seeds = @(3400, 3401, 3402, 3403, 3404)
    prohibited_registered_held_out_seeds = @(3100, 3101, 3102, 3103, 3104)
    episodes = 500
    max_steps = 300
    architecture = "equivariant_set_branching"
    observation_schema = "phase2d_ttl_cap_v2"
    projection_budget = 16
    environment_profile = "hta-mac/config/paper_aligned_hasani2025_b16.json"
    environment_profile_sha256 = (Get-FileHash -LiteralPath $Profile -Algorithm SHA256).Hash.ToLowerInvariant()
    qos_constraint_config = "hta-mac/config/paper_aligned_hasani2025_qos_constraints.json"
    qos_constraint_sha256 = (Get-FileHash -LiteralPath $Constraints -Algorithm SHA256).Hash.ToLowerInvariant()
    reward_scale = "generated_in_notebook_from_development_rollouts"
    held_out_seeds_used = $false
    files = $files
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $PackedStage2 "COLAB_PAPER_ALIGNED_B16_MANIFEST.json") -Encoding utf8

$zip = Join-Path $OutputDirectory "HTA_MAC_PaperAligned_B16_Training_Bundle_20260806.zip"
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
