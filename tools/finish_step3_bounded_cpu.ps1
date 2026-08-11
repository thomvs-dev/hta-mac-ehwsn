param(
    [Parameter(Mandatory = $true)]
    [int]$ProcessId,
    [int]$SampleSeconds = 10,
    [int]$CpuThreads = 18
)

$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$RunName = "step3_v3_bounded_100ep_seed5599_cpu$CpuThreads"
$RunDir = Join-Path $Repo "outputs\phase2\$RunName"
$ExportDir = Join-Path $Repo "outputs\local_cpu_export\$RunName"
$UtilizationReport = Join-Path $ExportDir 'cpu_utilization.json'
$CheckpointGate = Join-Path $ExportDir 'STEP3_BOUNDED_CHECKPOINT_GATE.json'
$LogicalProcessors = [Environment]::ProcessorCount
New-Item -ItemType Directory -Force -Path $ExportDir | Out-Null

$Process = Get-Process -Id $ProcessId -ErrorAction Stop
$InitialCpu = $Process.TotalProcessorTime.TotalSeconds
$InitialTime = [DateTime]::UtcNow
$PreviousCpu = $InitialCpu
$PreviousTime = $InitialTime
$Samples = [System.Collections.Generic.List[object]]::new()
$LastEpisode = -1
Write-Host "ATTACHED_PID=$ProcessId RUN_NAME=$RunName"

while ($true) {
    Start-Sleep -Seconds $SampleSeconds
    $Process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($null -eq $Process) { break }
    $Now = [DateTime]::UtcNow
    $Cpu = $Process.TotalProcessorTime.TotalSeconds
    $WallDelta = ($Now - $PreviousTime).TotalSeconds
    $CpuDelta = $Cpu - $PreviousCpu
    $Utilization = if ($WallDelta -gt 0) {
        100.0 * $CpuDelta / ($WallDelta * $LogicalProcessors)
    } else { 0.0 }
    $Samples.Add([pscustomobject]@{
        timestamp_utc = $Now.ToString('o')
        total_machine_cpu_percent = [Math]::Round($Utilization, 3)
        process_cpu_seconds_delta = [Math]::Round($CpuDelta, 3)
        wall_seconds_delta = [Math]::Round($WallDelta, 3)
    })
    $EpisodesPath = Join-Path $RunDir 'episodes.jsonl'
    if (Test-Path $EpisodesPath) {
        $Episode = (Get-Content $EpisodesPath | Measure-Object -Line).Lines
        if ($Episode -ne $LastEpisode) {
            Write-Host ("EPISODES={0} CPU_TOTAL_CAPACITY={1:N1}%" -f $Episode, $Utilization)
            $LastEpisode = $Episode
        }
    }
    $PreviousCpu = $Cpu
    $PreviousTime = $Now
}

$EndTime = [DateTime]::UtcNow
$Values = @($Samples | ForEach-Object { $_.total_machine_cpu_percent })
$Report = [ordered]@{
    schema_version = 1
    run_name = $RunName
    pid = $ProcessId
    cpu_threads = $CpuThreads
    logical_processors = $LogicalProcessors
    monitoring_started_utc = $InitialTime.ToString('o')
    monitoring_ended_utc = $EndTime.ToString('o')
    utilization_denominator = 'all_logical_processors'
    useful_training_process_only = $true
    mean_monitored_total_machine_cpu_percent = if ($Values.Count) { [Math]::Round(($Values | Measure-Object -Average).Average, 3) } else { 0.0 }
    peak_monitored_total_machine_cpu_percent = if ($Values.Count) { [Math]::Round(($Values | Measure-Object -Maximum).Maximum, 3) } else { 0.0 }
    samples = $Samples
}
$Report | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $UtilizationReport

$Checkpoint = Join-Path $ExportDir 'stability_episode_100.pt'
$Episodes = Join-Path $ExportDir 'episodes.jsonl'
if (-not (Test-Path $Checkpoint) -or -not (Test-Path $Episodes)) {
    throw "Trainer stopped before episode-100 artifacts were exported."
}
$Python = (Get-Command python).Source
Push-Location $Repo
try {
    & $Python -B validation/analyze_step3_bounded_training_checkpoint.py `
        --checkpoint $Checkpoint --episodes-jsonl $Episodes --output $CheckpointGate
    $GateExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
Write-Host "CPU_REPORT=$UtilizationReport"
Write-Host "CHECKPOINT_GATE=$CheckpointGate"
if ($GateExitCode -notin @(0, 3)) { exit $GateExitCode }
exit $GateExitCode
