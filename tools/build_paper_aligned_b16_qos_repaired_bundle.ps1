param([string]$OutputDirectory = (Join-Path $PSScriptRoot "..\colab"))

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Stage2 = (Resolve-Path (Join-Path $Repo "..")).Path
$FinalRepo = Join-Path $Stage2 "final_repo"
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$BuildRoot = Join-Path $OutputDirectory "_paper_aligned_b16_qos_repaired_build"
$Expected = [IO.Path]::GetFullPath((Join-Path $Repo "colab\_paper_aligned_b16_qos_repaired_build"))
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
    "HTA_MAC_PAPER_ALIGNED_B16_TRAINED_RESULTS_ANALYSIS_20260807.md",
    "HTA_MAC_PAPER_COMPARISON_AND_QOS_REPAIR_IMPLEMENTATION_20260808.md",
    "HTA_MAC_QOS_REPAIRED_AGENT_EXECUTION_HANDOFF_20260808.md",
    "HTA_MAC_EXTERNAL_REVIEW_RESOLUTION_AND_PREFLIGHT_DECISION_20260808.md"
)) {
    if (Test-Path -LiteralPath (Join-Path $Repo $file)) {
        Copy-Item -LiteralPath (Join-Path $Repo $file) -Destination $PackedHTA -Force
    }
}

# The complete validation suite expects this frozen shared-contract checkpoint.
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

$PackedPreflight = Join-Path $PackedHTA "outputs\preflight_20260808"
New-Item -ItemType Directory -Force -Path $PackedPreflight | Out-Null
$PreflightFiles = @(
    (Join-Path $Repo "outputs\audits\paper_aligned_b16_current_code_preflight_foundation_seed5299_20260808.json"),
    (Join-Path $Repo "outputs\phase3\paper_aligned_b16_budget_pressure_audit_seed5299_20260808\summary.json"),
    (Join-Path $Repo "outputs\phase3\paper_aligned_b16_budget_pressure_audit_seed5299_20260808\raw_trials.csv")
)
foreach ($source in $PreflightFiles) {
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Required preflight evidence is missing: $source"
    }
    Copy-Item -LiteralPath $source -Destination $PackedPreflight -Force
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
$Profile = Join-Path $Repo "config\paper_aligned_hasani2025_b16_qos_repaired.json"
$Constraints = Join-Path $Repo "config\paper_aligned_hasani2025_qos_constraints_repaired.json"
$GlobalGates = Join-Path $Repo "config\paper_aligned_hasani2025_global_evaluation_gates_repaired.json"
$ArchitectureDecision = Join-Path $Repo "config\paper_aligned_hasani2025_architecture_decision_repaired.json"
$PreflightFoundation = Join-Path $Repo "outputs\audits\paper_aligned_b16_current_code_preflight_foundation_seed5299_20260808.json"
$SourceCommit = (git -C $Stage2 rev-parse HEAD).Trim()
$SourceStatus = @(git -C $Stage2 status --short -- $Repo)
$manifest = [ordered]@{
    schema_version = 2
    purpose = "HTA-MAC post-repair paper-aligned B16 training and long-horizon development evaluation"
    claim_boundary = "paper_aligned_not_a_third_party_reproduction"
    created_utc = [DateTime]::UtcNow.ToString("o")
    track_role = "secondary_literature_alignment_side_study"
    primary_track_replaced = $false
    primary_contributions_evaluated_by_this_profile = $false
    stage2_source_commit = $SourceCommit
    hta_mac_source_status = $SourceStatus
    frozen_heart_ch_commit = (git -C $FinalRepo rev-parse HEAD).Trim()
    optimizer_seeds = @(5399, 6399, 7399)
    development_seeds = @(2400, 2401, 2402, 2403, 2404)
    reserved_confirmation_seeds = @(3400, 3401, 3402, 3403, 3404)
    prohibited_registered_held_out_seeds = @(3100, 3101, 3102, 3103, 3104)
    episodes = 500
    training_horizon = 300
    long_horizon = 3000
    architecture = "equivariant_set_branching"
    observation_schema = "phase2d_ttl_cap_v2"
    projection_budget = 16
    architecture_decision = "hta-mac/config/paper_aligned_hasani2025_architecture_decision_repaired.json"
    architecture_decision_sha256 = (Get-FileHash -LiteralPath $ArchitectureDecision -Algorithm SHA256).Hash.ToLowerInvariant()
    current_code_preflight_foundation_audit = "hta-mac/outputs/preflight_20260808/paper_aligned_b16_current_code_preflight_foundation_seed5299_20260808.json"
    current_code_preflight_foundation_sha256 = (Get-FileHash -LiteralPath $PreflightFoundation -Algorithm SHA256).Hash.ToLowerInvariant()
    environment_profile = "hta-mac/config/paper_aligned_hasani2025_b16_qos_repaired.json"
    environment_profile_sha256 = (Get-FileHash -LiteralPath $Profile -Algorithm SHA256).Hash.ToLowerInvariant()
    qos_constraint_config = "hta-mac/config/paper_aligned_hasani2025_qos_constraints_repaired.json"
    qos_constraint_sha256 = (Get-FileHash -LiteralPath $Constraints -Algorithm SHA256).Hash.ToLowerInvariant()
    global_evaluation_gates = "hta-mac/config/paper_aligned_hasani2025_global_evaluation_gates_repaired.json"
    global_evaluation_gates_sha256 = (Get-FileHash -LiteralPath $GlobalGates -Algorithm SHA256).Hash.ToLowerInvariant()
    phase3_reports_budget_utilization = $true
    phase3_reports_feasible_demand_contention = $true
    reward_scale = "generated_in_notebook_from_development_rollouts"
    held_out_seeds_used = $false
    files = $files
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $PackedStage2 "COLAB_PAPER_ALIGNED_B16_QOS_REPAIRED_MANIFEST.json") -Encoding utf8

$zip = Join-Path $OutputDirectory "HTA_MAC_PaperAligned_B16_QoSRepaired_Training_Bundle_20260808.zip"
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
