param([int]$CpuThreads = 18, [int]$SampleSeconds = 10)

$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$LogicalProcessors = [Environment]::ProcessorCount
if ($CpuThreads -lt 1 -or $CpuThreads -gt $LogicalProcessors) { throw 'Invalid CPU thread count.' }
$RunName = "step3_qos_deficit_bounded_100ep_seed5599_cpu$CpuThreads"
$ExportDir = Join-Path $Repo "outputs\local_cpu_export\$RunName"
$LogDir = Join-Path $Repo 'outputs\logs'
$StdoutLog = Join-Path $LogDir "$RunName.stdout.log"
$StderrLog = Join-Path $LogDir "$RunName.stderr.log"
$CpuReport = Join-Path $ExportDir 'cpu_utilization.json'
$GateReport = Join-Path $ExportDir 'STEP3_BOUNDED_CHECKPOINT_GATE.json'
New-Item -ItemType Directory -Force -Path $ExportDir, $LogDir | Out-Null

$Python = (Get-Command python).Source
$ProcessPath = [Environment]::GetEnvironmentVariable('Path', 'Process')
[Environment]::SetEnvironmentVariable('PATH', $null, 'Process')
[Environment]::SetEnvironmentVariable('Path', $ProcessPath, 'Process')
$Arguments = @(
    '-B', '-m', 'experiments.train_step3_qos_deficit_cpu',
    '--cpu-threads', "$CpuThreads",
    '--qos-deficit-config', 'config/step3_qos_deficit_override_selected_v2.json',
    '--ch-risk-config', 'config/step3_v3_risk_weight_5.json',
    '--step3-qos-config', 'config/step3_v3_qos_ema_floor_candidate.json',
    '--runtime-contract', 'outputs/audits/STEP3_LOCAL_CPU_RUNTIME_CONTRACT_FINAL_20260810.json',
    '--preflight-report', 'outputs/audits/STEP3_LOCAL_CPU_PREFLIGHT_20260810.json',
    '--checkpoint-export-dir', $ExportDir,
    '--episodes', '100', '--max-steps', '1200', '--development-seeds', '2400',
    '--optimizer-seed', '5599', '--run-name', $RunName,
    '--architecture', 'equivariant_set_branching', '--projection-budget', '16',
    '--reward-scale-config', 'outputs/audits/STEP3_LOCAL_CPU_RETURN_SCALE_20260810.json',
    '--environment-profile', 'config/paper_aligned_hasani2025_b16_qos_repaired.json',
    '--normalize-input-blocks', '--learning-rate', '1e-5',
    '--trajectory-loss-weight', '1.0', '--concavity-loss-weight', '0.1',
    '--learn-every', '4', '--precision', 'fp32',
    '--stability-interval', '25', '--stability-tail-episodes', '100', '--device', 'cpu'
)
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Repo `
    -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog `
    -WindowStyle Hidden -PassThru
try { $Process.PriorityClass = 'AboveNormal' } catch { Write-Warning $_ }
Write-Host "RUN_NAME=$RunName PID=$($Process.Id) CPU_THREADS=$CpuThreads"

$Samples = [System.Collections.Generic.List[object]]::new()
$PreviousCpu = $Process.TotalProcessorTime.TotalSeconds
$PreviousTime = [DateTime]::UtcNow
while (-not $Process.HasExited) {
    Start-Sleep -Seconds $SampleSeconds
    $Process.Refresh()
    if ($Process.HasExited) { break }
    $Now = [DateTime]::UtcNow
    $Cpu = $Process.TotalProcessorTime.TotalSeconds
    $Wall = ($Now - $PreviousTime).TotalSeconds
    $Util = if ($Wall -gt 0) { 100.0 * ($Cpu - $PreviousCpu) / ($Wall * $LogicalProcessors) } else { 0.0 }
    if ($Util -ge 0.0 -and $Util -le 100.0) {
        $Samples.Add([pscustomobject]@{timestamp_utc=$Now.ToString('o');total_machine_cpu_percent=[Math]::Round($Util,3)})
    }
    $EpisodesPath = Join-Path $Repo "outputs\phase2\$RunName\episodes.jsonl"
    if (Test-Path $EpisodesPath) {
        $Count = (Get-Content $EpisodesPath | Measure-Object -Line).Lines
        Write-Host ("EPISODES={0} CPU_TOTAL_CAPACITY={1:N1}%" -f $Count, $Util)
    }
    $PreviousCpu, $PreviousTime = $Cpu, $Now
}
$Process.WaitForExit()
$TrainingExit = $Process.ExitCode
$Values = @($Samples | ForEach-Object { $_.total_machine_cpu_percent })
[ordered]@{
    schema_version=1; run_name=$RunName; pid=$Process.Id; cpu_threads=$CpuThreads;
    logical_processors=$LogicalProcessors; useful_training_process_only=$true;
    mean_total_machine_cpu_percent=if($Values.Count){[Math]::Round(($Values|Measure-Object -Average).Average,3)}else{0.0};
    peak_total_machine_cpu_percent=if($Values.Count){[Math]::Round(($Values|Measure-Object -Maximum).Maximum,3)}else{0.0};
    samples=$Samples; training_exit_code=$TrainingExit
} | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $CpuReport

$Checkpoint = Join-Path $ExportDir 'stability_episode_100.pt'
$Episodes = Join-Path $ExportDir 'episodes.jsonl'
if (-not (Test-Path $Checkpoint) -or -not (Test-Path $Episodes)) {
    throw "Episode-100 artifacts missing; inspect $StdoutLog and $StderrLog."
}
Push-Location $Repo
try {
    & $Python -B validation/analyze_step3_bounded_training_checkpoint.py `
        --checkpoint $Checkpoint --episodes-jsonl $Episodes --output $GateReport
    $GateExit = $LASTEXITCODE
} finally { Pop-Location }
Write-Host "TRAINING_EXIT=$TrainingExit GATE_EXIT=$GateExit CPU_REPORT=$CpuReport GATE=$GateReport"
if ($TrainingExit -notin @(0,3)) { exit $TrainingExit }
if ($GateExit -notin @(0,3)) { exit $GateExit }
exit $GateExit
