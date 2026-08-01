param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\colab")
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Stage2 = (Resolve-Path (Join-Path $Repo "..")).Path
$FinalRepo = Join-Path $Stage2 "final_repo"
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$BuildRoot = Join-Path $OutputDirectory "_bundle_build"
$ExpectedBuildRoot = [IO.Path]::GetFullPath((Join-Path $Repo "colab\_bundle_build"))
if ([IO.Path]::GetFullPath($BuildRoot) -ne $ExpectedBuildRoot) {
    throw "Refusing to clean unexpected build path: $BuildRoot"
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
Copy-Item -LiteralPath (Join-Path $Repo ".git") -Destination $PackedHTA -Recurse -Force
foreach ($file in @(
    ".gitignore",
    "README.md",
    "PHASE4_PREREGISTRATION.md",
    "PHASE4_PREREGISTRATION.sha256",
    "pyproject.toml"
)) {
    if (Test-Path -LiteralPath (Join-Path $Repo $file)) {
        Copy-Item -LiteralPath (Join-Path $Repo $file) -Destination $PackedHTA -Force
    }
}

$PackedCache = Join-Path $PackedHTA "outputs\cache\phase3_schedules"
$PackedPhase2 = Join-Path $PackedHTA "outputs\phase2"
New-Item -ItemType Directory -Force -Path $PackedCache, $PackedPhase2 | Out-Null
foreach ($seed in 2300..2304) {
    $pattern = "seed_" + $seed + "_horizon_300_v2_*.pkl"
    $matches = @(Get-ChildItem -LiteralPath (Join-Path $Repo "outputs\cache\phase3_schedules") -File |
        Where-Object { $_.Name -like $pattern })
    if ($matches.Count -ne 1) {
        throw "Expected one schema-v2 cache for seed $seed, found $($matches.Count)"
    }
    Copy-Item -LiteralPath $matches[0].FullName -Destination $PackedCache -Force
}

foreach ($folder in @("configs", "ehwsn", "env", "models", "src", "utils")) {
    Copy-Item -LiteralPath (Join-Path $FinalRepo $folder) -Destination $PackedFinal -Recurse -Force
}
Copy-Item -LiteralPath (Join-Path $FinalRepo ".git") -Destination $PackedFinal -Recurse -Force
Get-ChildItem -LiteralPath $FinalRepo -File |
    Where-Object { $_.Extension -in @(".py", ".toml", ".txt") } |
    Copy-Item -Destination $PackedFinal -Force

$PackedFinalOutputs = Join-Path $PackedFinal "outputs"
$PackedCheckpoints = Join-Path $PackedFinalOutputs "checkpoints"
New-Item -ItemType Directory -Force -Path $PackedCheckpoints | Out-Null
Copy-Item -LiteralPath (Join-Path $FinalRepo "outputs\stage1_params.mat") -Destination $PackedFinalOutputs -Force
Copy-Item -LiteralPath (Join-Path $FinalRepo "outputs\checkpoints\model_v91_throughput.pt") -Destination $PackedCheckpoints -Force

$sourceCommit = (git -C $Repo rev-parse HEAD).Trim()
$upstreamCommit = (git -C $FinalRepo rev-parse HEAD).Trim()
$files = @(
    Get-ChildItem -LiteralPath $PackedStage2 -Recurse -File | ForEach-Object {
        [ordered]@{
            path = $_.FullName.Substring($PackedStage2.Length + 1).Replace("\", "/")
            bytes = $_.Length
            sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
)
$manifest = [ordered]@{
    schema_version = 1
    created_utc = [DateTime]::UtcNow.ToString("o")
    hta_mac_git_commit = $sourceCommit
    frozen_heart_ch_git_commit = $upstreamCommit
    registered_budgets = @(8, 12, 16, 20, 24)
    registered_training_seeds = @(2299, 3299, 4299)
    development_schedule_seeds = @(2300, 2301, 2302, 2303, 2304)
    episodes = 500
    max_steps = 300
    files = $files
}
$manifestPath = Join-Path $PackedStage2 "COLAB_BUNDLE_MANIFEST.json"
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding utf8

$readmeSource = Join-Path $Repo "colab\README.md"
Copy-Item -LiteralPath $readmeSource -Destination (Join-Path $PackedStage2 "COLAB_BUNDLE_README.md") -Force

$zip = Join-Path $OutputDirectory "HTA_MAC_Colab_Training_Bundle_20260801.zip"
if (Test-Path -LiteralPath $zip) {
    Remove-Item -LiteralPath $zip -Force
}
Compress-Archive -LiteralPath $PackedStage2 -DestinationPath $zip -CompressionLevel Optimal
$zipHash = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
"$zipHash  $([IO.Path]::GetFileName($zip))" |
    Set-Content -LiteralPath "$zip.sha256" -Encoding ascii

Write-Output "BUNDLE=$zip"
Write-Output "BYTES=$((Get-Item -LiteralPath $zip).Length)"
Write-Output "SHA256=$zipHash"
Write-Output "FILES=$($files.Count)"
Write-Output "HTA_COMMIT=$sourceCommit"
Write-Output "UPSTREAM_COMMIT=$upstreamCommit"

Remove-Item -LiteralPath $BuildRoot -Recurse -Force
