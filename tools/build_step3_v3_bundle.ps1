param([string]$OutputDirectory = (Join-Path $PSScriptRoot "..\colab"))

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Stage2 = (Resolve-Path (Join-Path $Repo "..")).Path
$FinalRepo = Join-Path $Stage2 "final_repo"
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$BuildRoot = Join-Path $OutputDirectory "_step3_v3_build"
$Expected = [IO.Path]::GetFullPath((Join-Path $Repo "colab\_step3_v3_build"))
if ([IO.Path]::GetFullPath($BuildRoot) -ne $Expected) { throw "Unexpected build root: $BuildRoot" }
if (Test-Path -LiteralPath $BuildRoot) { Remove-Item -LiteralPath $BuildRoot -Recurse -Force }

$PackedStage2 = Join-Path $BuildRoot "stage2"
$PackedHTA = Join-Path $PackedStage2 "hta-mac"
$PackedFinal = Join-Path $PackedStage2 "final_repo"
New-Item -ItemType Directory -Force -Path $PackedHTA, $PackedFinal | Out-Null
foreach ($folder in @("agents", "baselines", "config", "core", "envs", "experiments", "validation")) {
    Copy-Item -LiteralPath (Join-Path $Repo $folder) -Destination $PackedHTA -Recurse -Force
}
foreach ($file in @("README.md", "STEP3_RECOVERY_AND_QOS_IMPROVEMENT_IMPLEMENTATION_PLAN_20260809.md")) {
    $source = Join-Path $Repo $file
    if (Test-Path -LiteralPath $source -PathType Leaf) { Copy-Item -LiteralPath $source -Destination $PackedHTA -Force }
}

# Complete validation suite fixture.
$Authoritative = Join-Path $Repo "outputs\phase2\authoritative_dynamic_budget8_500ep"
$PackedAuthoritative = Join-Path $PackedHTA "outputs\phase2\authoritative_dynamic_budget8_500ep"
New-Item -ItemType Directory -Force -Path $PackedAuthoritative | Out-Null
foreach ($file in @("branching_c51.pt", "summary.json")) {
    $source = Join-Path $Authoritative $file
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing validation fixture $source" }
    Copy-Item -LiteralPath $source -Destination $PackedAuthoritative -Force
}

# Frozen Step-2 development comparator; it is never used to initialize v3.
$Reference = Join-Path $Repo "HTA_MAC_PaperAligned_B16_QoSRepaired_Trained_Results_20260808\phase2\paper_aligned_b16_qos_repaired_500ep_seed7399"
$PackedReference = Join-Path $PackedStage2 "reference_seed7399"
New-Item -ItemType Directory -Force -Path $PackedReference | Out-Null
foreach ($file in @("branching_c51.pt", "summary.json")) {
    $source = Join-Path $Reference $file
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing Step-2 reference $source" }
    Copy-Item -LiteralPath $source -Destination $PackedReference -Force
}

foreach ($folder in @("configs", "ehwsn", "env", "models", "src", "utils")) {
    Copy-Item -LiteralPath (Join-Path $FinalRepo $folder) -Destination $PackedFinal -Recurse -Force
}
Get-ChildItem -LiteralPath $FinalRepo -File | Where-Object { $_.Extension -in @(".py", ".toml", ".txt") } | Copy-Item -Destination $PackedFinal -Force
$PackedOutputs = Join-Path $PackedFinal "outputs"
$PackedCheckpoints = Join-Path $PackedOutputs "checkpoints"
New-Item -ItemType Directory -Force -Path $PackedCheckpoints | Out-Null
Copy-Item -LiteralPath (Join-Path $FinalRepo "outputs\stage1_params.mat") -Destination $PackedOutputs -Force
Copy-Item -LiteralPath (Join-Path $FinalRepo "outputs\checkpoints\model_v91_throughput.pt") -Destination $PackedCheckpoints -Force

$Caches = @(Get-ChildItem -LiteralPath $PackedStage2 -Directory -Recurse | Where-Object { $_.Name -eq "__pycache__" } | Sort-Object { $_.FullName.Length } -Descending)
foreach ($cache in $Caches) {
    if (-not $cache.FullName.StartsWith($PackedStage2, [StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe cache target" }
    Remove-Item -LiteralPath $cache.FullName -Recurse -Force
}
$files = @(Get-ChildItem -LiteralPath $PackedStage2 -Recurse -File | ForEach-Object {
    [ordered]@{
        path = $_.FullName.Substring($PackedStage2.Length + 1).Replace("\", "/")
        bytes = $_.Length
        sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
})
$manifest = [ordered]@{
    schema_version = 1
    purpose = "Step 3 v3 CH-aware QoS/lifetime development training"
    claim_boundary = "development_only_not_publication_or_confirmation_evidence"
    created_utc = [DateTime]::UtcNow.ToString("o")
    optimizer_seeds = @(5599, 6599, 7599)
    development_seeds = @(2400, 2401, 2402, 2403, 2404)
    prohibited_registered_seeds = @(3100, 3101, 3102, 3103, 3104)
    reserved_confirmation_seeds = @(3400, 3401, 3402, 3403, 3404)
    episodes = 500
    training_horizon = 1200
    evaluation_horizon = 3000
    observation_schema = "step3_ch_context_v3"
    observation_features = 65
    embedding_start = 33
    architecture = "equivariant_set_branching"
    projection_budget = 16
    learned_intervention = "mac_allocation_only"
    ch_schedule_modified = $false
    exact_same_runtime_required = $true
    cross_platform_tolerance_validated = $false
    least_bad_infeasible_selection_forbidden = $true
    files = $files
}
$ManifestPath = Join-Path $PackedStage2 "COLAB_STEP3_V3_MANIFEST.json"
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ManifestPath -Encoding utf8
$zip = Join-Path $OutputDirectory "HTA_MAC_Step3_V3_Training_Bundle_20260809.zip"
if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
Compress-Archive -LiteralPath $PackedStage2 -DestinationPath $zip -CompressionLevel Optimal
$hash = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
"$hash  $([IO.Path]::GetFileName($zip))" | Set-Content -LiteralPath "$zip.sha256" -Encoding ascii
$notebook = Join-Path $OutputDirectory "HTA_MAC_Step3_V3_Training_Colab_20260809.ipynb"
python -B (Join-Path $Repo "tools\generate_step3_v3_colab_final.py") --bundle-sha256 $hash --output $notebook
if ($LASTEXITCODE -ne 0) { throw "Notebook generation failed" }
Write-Output "BUNDLE=$zip"
Write-Output "SHA256=$hash"
Write-Output "NOTEBOOK=$notebook"
Write-Output "FILES=$($files.Count)"
Remove-Item -LiteralPath $BuildRoot -Recurse -Force
