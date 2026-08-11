param([int]$CpuThreads = 18, [int]$SampleSeconds = 5)

$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$LogicalProcessors = [Environment]::ProcessorCount
if ($CpuThreads -lt 1 -or $CpuThreads -gt $LogicalProcessors) { throw 'Invalid CPU thread count.' }
$RunName = "step3_dqfd_shield_distillation_seed5701_cpu$CpuThreads"
$OutputDir = Join-Path $Repo "outputs\phase2\$RunName"
$LogDir = Join-Path $Repo 'outputs\logs'
$StdoutLog = Join-Path $LogDir "$RunName.stdout.log"
$StderrLog = Join-Path $LogDir "$RunName.stderr.log"
$CpuReport = Join-Path $OutputDir 'cpu_utilization.json'
New-Item -ItemType Directory -Force -Path $OutputDir, $LogDir | Out-Null

$Python = (Get-Command python).Source
$ProcessPath = [Environment]::GetEnvironmentVariable('Path', 'Process')
[Environment]::SetEnvironmentVariable('PATH', $null, 'Process')
[Environment]::SetEnvironmentVariable('Path', $ProcessPath, 'Process')
$Arguments = @(
    '-B', '-m', 'experiments.distill_step3_qos_shield',
    '--checkpoint', 'outputs/local_cpu_export/step3_qos_deficit_bounded_100ep_seed5599_cpu18/stability_episode_100.pt',
    '--environment-profile', 'config/paper_aligned_hasani2025_b16_qos_repaired.json',
    '--ch-risk-config', 'config/step3_v3_risk_weight_5.json',
    '--qos-config', 'config/step3_v3_qos_ema_floor_candidate.json',
    '--controller-config', 'config/step3_qos_deficit_override_selected_v2.json',
    '--output-dir', $OutputDir,
    '--epochs', '5', '--batch-size', '64', '--learning-rate', '3e-5',
    '--margin', '0.8', '--seed', '5701', '--cpu-threads', "$CpuThreads", '--horizon', '1200'
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
    Write-Host ("CPU_TOTAL_CAPACITY={0:N1}%" -f $Util)
    $PreviousCpu, $PreviousTime = $Cpu, $Now
}
$Process.WaitForExit()
$Process.Refresh()
$Values = @($Samples | ForEach-Object { $_.total_machine_cpu_percent })
[ordered]@{
    schema_version=1; run_name=$RunName; pid=$Process.Id; cpu_threads=$CpuThreads;
    logical_processors=$LogicalProcessors; useful_training_process_only=$true;
    mean_total_machine_cpu_percent=if($Values.Count){[Math]::Round(($Values|Measure-Object -Average).Average,3)}else{0.0};
    peak_total_machine_cpu_percent=if($Values.Count){[Math]::Round(($Values|Measure-Object -Maximum).Maximum,3)}else{0.0};
    samples=$Samples; exit_code=$Process.ExitCode
} | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $CpuReport
Write-Host "EXIT=$($Process.ExitCode) CPU_REPORT=$CpuReport"
exit $Process.ExitCode
