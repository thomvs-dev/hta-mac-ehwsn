param(
    [int]$CpuThreads = 18,
    [int]$SampleSeconds = 10
)

$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$LogicalProcessors = [Environment]::ProcessorCount
if ($CpuThreads -lt 1 -or $CpuThreads -gt $LogicalProcessors) {
    throw "CpuThreads must be between 1 and $LogicalProcessors."
}

$RunName = "step3_v3_bounded_100ep_seed5599_cpu$CpuThreads"
$ExportDir = Join-Path $Repo "outputs\local_cpu_export\$RunName"
$LogDir = Join-Path $Repo 'outputs\logs'
$StdoutLog = Join-Path $LogDir "$RunName.stdout.log"
$StderrLog = Join-Path $LogDir "$RunName.stderr.log"
$UtilizationReport = Join-Path $ExportDir 'cpu_utilization.json'
$CheckpointGate = Join-Path $ExportDir 'STEP3_BOUNDED_CHECKPOINT_GATE.json'
New-Item -ItemType Directory -Force -Path $ExportDir, $LogDir | Out-Null

$Python = (Get-Command python).Source
# Some managed shells inject both Path and PATH. Windows PowerShell's
# Start-Process rejects that case-colliding environment block.
$ProcessPath = [Environment]::GetEnvironmentVariable('Path', 'Process')
[Environment]::SetEnvironmentVariable('PATH', $null, 'Process')
[Environment]::SetEnvironmentVariable('Path', $ProcessPath, 'Process')
$Arguments = @(
    '-B', '-m', 'experiments.train_step3_v3_cpu',
    '--cpu-threads', "$CpuThreads",
    '--ch-risk-config', 'config/step3_v3_risk_weight_5.json',
    '--step3-qos-config', 'config/step3_v3_qos_ema_floor_candidate.json',
    '--runtime-contract', 'outputs/audits/STEP3_LOCAL_CPU_RUNTIME_CONTRACT_FINAL_20260810.json',
    '--preflight-report', 'outputs/audits/STEP3_LOCAL_CPU_PREFLIGHT_20260810.json',
    '--checkpoint-export-dir', $ExportDir,
    '--episodes', '100',
    '--max-steps', '1200',
    '--development-seeds', '2400',
    '--optimizer-seed', '5599',
    '--run-name', $RunName,
    '--architecture', 'equivariant_set_branching',
    '--projection-budget', '16',
    '--reward-scale-config', 'outputs/audits/STEP3_LOCAL_CPU_RETURN_SCALE_20260810.json',
    '--environment-profile', 'config/paper_aligned_hasani2025_b16_qos_repaired.json',
    '--normalize-input-blocks',
    '--learning-rate', '1e-5',
    '--trajectory-loss-weight', '1.0',
    '--concavity-loss-weight', '0.1',
    '--learn-every', '4',
    '--precision', 'fp32',
    '--stability-interval', '25',
    '--stability-tail-episodes', '100',
    '--device', 'cpu'
)

$Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Repo `
    -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog `
    -WindowStyle Hidden -PassThru
try { $Process.PriorityClass = 'AboveNormal' } catch { Write-Warning $_ }

Write-Host "RUN_NAME=$RunName"
Write-Host "PID=$($Process.Id) CPU_THREADS=$CpuThreads LOGICAL_PROCESSORS=$LogicalProcessors"
Write-Host "STDOUT=$StdoutLog"

$Samples = [System.Collections.Generic.List[object]]::new()
$PreviousCpu = $Process.TotalProcessorTime.TotalSeconds
$PreviousTime = [DateTime]::UtcNow
$LastProgress = ''
while (-not $Process.HasExited) {
    Start-Sleep -Seconds $SampleSeconds
    $Process.Refresh()
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
    Write-Host ("CPU_TOTAL_CAPACITY={0:N1}% ELAPSED_MIN={1:N1}" -f $Utilization, (($Now - $Process.StartTime.ToUniversalTime()).TotalMinutes))
    if (Test-Path $StdoutLog) {
        $Progress = Get-Content $StdoutLog | Where-Object { $_ -match '^(EPISODE=|POLICY_STABILITY_SNAPSHOT=)' } | Select-Object -Last 1
        if ($Progress -and $Progress -ne $LastProgress) {
            Write-Host $Progress
            $LastProgress = $Progress
        }
    }
    $PreviousCpu = $Cpu
    $PreviousTime = $Now
}
$Process.WaitForExit()
$TrainingExitCode = $Process.ExitCode

$Values = @($Samples | ForEach-Object { $_.total_machine_cpu_percent })
$Report = [ordered]@{
    schema_version = 1
    run_name = $RunName
    pid = $Process.Id
    cpu_threads = $CpuThreads
    logical_processors = $LogicalProcessors
    utilization_denominator = 'all_logical_processors'
    useful_training_process_only = $true
    samples = $Samples
    mean_total_machine_cpu_percent = if ($Values.Count) { [Math]::Round(($Values | Measure-Object -Average).Average, 3) } else { 0.0 }
    peak_total_machine_cpu_percent = if ($Values.Count) { [Math]::Round(($Values | Measure-Object -Maximum).Maximum, 3) } else { 0.0 }
    training_exit_code = $TrainingExitCode
    stdout_log = $StdoutLog
    stderr_log = $StderrLog
}
$Report | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $UtilizationReport
Write-Host "TRAINING_EXIT_CODE=$TrainingExitCode"
Write-Host "CPU_REPORT=$UtilizationReport"

$Checkpoint = Join-Path $ExportDir 'stability_episode_100.pt'
$Episodes = Join-Path $ExportDir 'episodes.jsonl'
if (-not (Test-Path $Checkpoint) -or -not (Test-Path $Episodes)) {
    throw "Episode-100 artifacts were not persisted. Inspect $StdoutLog and $StderrLog."
}

& $Python -B validation/analyze_step3_bounded_training_checkpoint.py `
    --checkpoint $Checkpoint --episodes-jsonl $Episodes --output $CheckpointGate
$GateExitCode = $LASTEXITCODE
Write-Host "CHECKPOINT_GATE_EXIT_CODE=$GateExitCode"
Write-Host "CHECKPOINT_GATE=$CheckpointGate"
if ($TrainingExitCode -notin @(0, 3)) { exit $TrainingExitCode }
if ($GateExitCode -notin @(0, 3)) { exit $GateExitCode }
exit $GateExitCode
