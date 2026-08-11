param([string]$OutputDirectory = (Join-Path $PSScriptRoot "..\colab"))

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Stage2 = (Resolve-Path (Join-Path $Repo "..")).Path
$FinalRepo = Join-Path $Stage2 "final_repo"
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$BuildRoot = Join-Path $OutputDirectory "_step3_ch_role_lifetime_build"
$Expected = [IO.Path]::GetFullPath((Join-Path $Repo "colab\_step3_ch_role_lifetime_build"))
if ([IO.Path]::GetFullPath($BuildRoot) -ne $Expected) { throw "Unexpected build root: $BuildRoot" }
if (Test-Path -LiteralPath $BuildRoot) { Remove-Item -LiteralPath $BuildRoot -Recurse -Force }

$PackedStage2 = Join-Path $BuildRoot "stage2"
$PackedHTA = Join-Path $PackedStage2 "hta-mac"
$PackedFinal = Join-Path $PackedStage2 "final_repo"
New-Item -ItemType Directory -Force -Path $PackedHTA, $PackedFinal | Out-Null
foreach ($folder in @("agents", "baselines", "config", "core", "envs", "experiments", "validation")) {
    Copy-Item -LiteralPath (Join-Path $Repo $folder) -Destination $PackedHTA -Recurse -Force
}
foreach ($file in @("README.md", "STEP3_CH_ROLE_LIFETIME_EXECUTION_CONTRACT_20260808.md", "QOS_REPAIRED_REVIEW_RESPONSE_AND_STEP3_DECISION_20260808.md")) {
    $source = Join-Path $Repo $file
    if (Test-Path -LiteralPath $source) { Copy-Item -LiteralPath $source -Destination $PackedHTA -Force }
}

# Frozen checkpoint required by the complete shared-policy validation suite.
$Authoritative = Join-Path $Repo "outputs\phase2\authoritative_dynamic_budget8_500ep"
$PackedAuthoritative = Join-Path $PackedHTA "outputs\phase2\authoritative_dynamic_budget8_500ep"
New-Item -ItemType Directory -Force -Path $PackedAuthoritative | Out-Null
foreach ($file in @("branching_c51.pt", "summary.json")) {
    $source = Join-Path $Authoritative $file
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing $source" }
    Copy-Item -LiteralPath $source -Destination $PackedAuthoritative -Force
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
$Risk = Join-Path $Repo "config\step3_ch_role_depletion_risk_v1.json"
$manifest = [ordered]@{
    schema_version = 1
    purpose = "Step 3 scheduled-CH depletion-risk MAC training"
    claim_boundary = "development_mechanism_candidate_not_lifetime_superiority"
    created_utc = [DateTime]::UtcNow.ToString("o")
    optimizer_seeds = @(5499, 6499, 7499)
    development_seeds = @(2400, 2401, 2402, 2403, 2404)
    reserved_confirmation_seeds = @(3400, 3401, 3402, 3403, 3404)
    prohibited_registered_held_out_seeds = @(3100, 3101, 3102, 3103, 3104)
    episodes = 500
    training_horizon = 1200
    evaluation_horizon = 3000
    architecture = "equivariant_set_branching"
    projection_budget = 16
    learned_intervention = "mac_allocation_only"
    ch_schedule_modified = $false
    role_separated_energy_accounting = $true
    exact_same_runtime_required = $true
    cross_platform_tolerance_validated = $false
    ch_risk_config = "hta-mac/config/step3_ch_role_depletion_risk_v1.json"
    ch_risk_config_sha256 = (Get-FileHash -LiteralPath $Risk -Algorithm SHA256).Hash.ToLowerInvariant()
    files = $files
}
$ManifestPath = Join-Path $PackedStage2 "COLAB_STEP3_CH_ROLE_LIFETIME_MANIFEST.json"
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ManifestPath -Encoding utf8
$zip = Join-Path $OutputDirectory "HTA_MAC_Step3_CHRole_Lifetime_Training_Bundle_20260808.zip"
if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
Compress-Archive -LiteralPath $PackedStage2 -DestinationPath $zip -CompressionLevel Optimal
$hash = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
"$hash  $([IO.Path]::GetFileName($zip))" | Set-Content -LiteralPath "$zip.sha256" -Encoding ascii
$notebook = Join-Path $OutputDirectory "HTA_MAC_Step3_CHRole_Lifetime_Training_Colab_20260808.ipynb"
python -B (Join-Path $Repo "tools\generate_step3_colab.py") --bundle-sha256 $hash --output $notebook
if ($LASTEXITCODE -ne 0) { throw "Notebook generation failed" }
Write-Output "BUNDLE=$zip"
Write-Output "SHA256=$hash"
Write-Output "NOTEBOOK=$notebook"
Write-Output "FILES=$($files.Count)"
Remove-Item -LiteralPath $BuildRoot -Recurse -Force
