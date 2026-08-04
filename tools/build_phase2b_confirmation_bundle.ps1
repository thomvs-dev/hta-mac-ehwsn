param([string]$OutputDirectory = (Join-Path $PSScriptRoot "..\colab"))
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Stage2 = (Resolve-Path (Join-Path $Repo "..")).Path
$FinalRepo = Join-Path $Stage2 "final_repo"
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$BuildRoot = Join-Path $OutputDirectory "_phase2b_bundle_build"
$Expected = [IO.Path]::GetFullPath((Join-Path $Repo "colab\_phase2b_bundle_build"))
if ([IO.Path]::GetFullPath($BuildRoot) -ne $Expected) {
    throw "Refusing unexpected build root: $BuildRoot"
}
if (Test-Path -LiteralPath $BuildRoot) {
    Remove-Item -LiteralPath $BuildRoot -Recurse -Force
}
$PackedStage2 = Join-Path $BuildRoot "stage2"
$PackedHTA = Join-Path $PackedStage2 "hta-mac"
$PackedFinal = Join-Path $PackedStage2 "final_repo"
New-Item -ItemType Directory -Force -Path $PackedHTA,$PackedFinal | Out-Null

foreach($folder in @("agents","baselines","config","core","envs","experiments","validation")){
    Copy-Item -LiteralPath (Join-Path $Repo $folder) -Destination $PackedHTA -Recurse -Force
}
foreach($file in @(".gitignore","README.md","HTA_MAC_PHASE2B_LOCAL_REPAIR_AND_PUBLICATION_DECISION_20260803.md")){
    Copy-Item -LiteralPath (Join-Path $Repo $file) -Destination $PackedHTA -Force
}
$PackedCache = Join-Path $PackedHTA "outputs\cache\phase3_schedules"
$PackedInputs = Join-Path $PackedHTA "inputs"
New-Item -ItemType Directory -Force -Path $PackedCache,$PackedInputs | Out-Null
foreach($seed in 2300..2304){
    $pattern="seed_" + $seed + "_horizon_300_v2_*.pkl"
    $matches=@(Get-ChildItem -LiteralPath (Join-Path $Repo "outputs\cache\phase3_schedules") -File |
        Where-Object {$_.Name -like $pattern})
    if($matches.Count -ne 1){throw "Expected one schedule cache for $seed, found $($matches.Count)"}
    Copy-Item -LiteralPath $matches[0].FullName -Destination $PackedCache -Force
}
foreach($seed in @(2299,3299,4299)){
    $relative="HTA_MAC_Phase2_Registered\runs\registered_shared_b12_seed" + $seed + "\branching_c51.pt"
    $source=Join-Path $Repo $relative
    if(-not (Test-Path -LiteralPath $source)){throw "Missing checkpoint: $source"}
    $destination=Join-Path $PackedInputs ("registered_shared_b12_seed" + $seed + ".pt")
    Copy-Item -LiteralPath $source -Destination $destination -Force
}

foreach($folder in @("configs","ehwsn","env","models","src","utils")){
    Copy-Item -LiteralPath (Join-Path $FinalRepo $folder) -Destination $PackedFinal -Recurse -Force
}
Get-ChildItem -LiteralPath $FinalRepo -File |
    Where-Object {$_.Extension -in @(".py",".toml",".txt")} |
    Copy-Item -Destination $PackedFinal -Force
$PackedFinalOutputs=Join-Path $PackedFinal "outputs"
$PackedCheckpoints=Join-Path $PackedFinalOutputs "checkpoints"
New-Item -ItemType Directory -Force -Path $PackedCheckpoints | Out-Null
Copy-Item -LiteralPath (Join-Path $FinalRepo "outputs\stage1_params.mat") -Destination $PackedFinalOutputs -Force
Copy-Item -LiteralPath (Join-Path $FinalRepo "outputs\checkpoints\model_v91_throughput.pt") -Destination $PackedCheckpoints -Force

$files=@(Get-ChildItem -LiteralPath $PackedStage2 -Recurse -File | ForEach-Object {
    [ordered]@{
        path=$_.FullName.Substring($PackedStage2.Length+1).Replace("\","/")
        bytes=$_.Length
        sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
})
$manifest=[ordered]@{
    schema_version=2
    purpose="HTA-MAC Phase 2B three-seed budget-12 confirmation"
    created_utc=[DateTime]::UtcNow.ToString("o")
    optimizer_seeds=@(2299,3299,4299)
    development_seeds=@(2300,2301,2302,2303,2304)
    episodes_default=125
    max_steps=300
    architecture="shared_branching"
    budget=12
    files=$files
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $PackedStage2 "COLAB_PHASE2B_MANIFEST.json") -Encoding utf8
$zip=Join-Path $OutputDirectory "HTA_MAC_Phase2B_Confirmation_Bundle_20260803.zip"
if(Test-Path -LiteralPath $zip){Remove-Item -LiteralPath $zip -Force}
Compress-Archive -LiteralPath $PackedStage2 -DestinationPath $zip -CompressionLevel Optimal
$hash=(Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
"$hash  $([IO.Path]::GetFileName($zip))" | Set-Content -LiteralPath "$zip.sha256" -Encoding ascii
Write-Output "BUNDLE=$zip"
Write-Output "BYTES=$((Get-Item -LiteralPath $zip).Length)"
Write-Output "SHA256=$hash"
Write-Output "FILES=$($files.Count)"
Remove-Item -LiteralPath $BuildRoot -Recurse -Force