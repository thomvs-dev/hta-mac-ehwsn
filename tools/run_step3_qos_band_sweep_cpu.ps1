param([int]$SampleSeconds = 5)

$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$RunName = 'step3_qos_band_projection_sweep_v1'
$OutputDir = Join-Path $Repo "outputs\audits\$RunName"
$LogDir = Join-Path $Repo 'outputs\logs'
New-Item -ItemType Directory -Force -Path $OutputDir, $LogDir | Out-Null
$StdoutLog = Join-Path $LogDir "$RunName.stdout.log"
$StderrLog = Join-Path $LogDir "$RunName.stderr.log"
$CpuReport = Join-Path $OutputDir 'cpu_utilization.json'
$SweepReport = Join-Path $OutputDir 'qos_band_sweep_report.json'
$Python = (Get-Command python).Source
$ProcessPath = [Environment]::GetEnvironmentVariable('Path', 'Process')
[Environment]::SetEnvironmentVariable('PATH', $null, 'Process')
[Environment]::SetEnvironmentVariable('Path', $ProcessPath, 'Process')
$Arguments = @(
    '-B', '-m', 'experiments.sweep_step3_qos_band_projection',
    '--checkpoint', 'outputs/phase2/step3_dqfd_shield_distillation_seed5701_cpu18/branching_c51_shield_distilled.pt',
    '--environment-profile', 'config/paper_aligned_hasani2025_b16_qos_repaired.json',
    '--ch-risk-config', 'config/step3_v3_risk_weight_5.json',
    '--qos-config', 'config/step3_v3_qos_ema_floor_candidate.json',
    '--sweep-config', 'config/step3_qos_band_projection_sweep_v1.json',
    '--output', $SweepReport
)
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Repo `
    -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog -WindowStyle Hidden -PassThru
try { $Process.PriorityClass = 'AboveNormal' } catch { Write-Warning $_ }
$LogicalProcessors = [Environment]::ProcessorCount
$Samples = [System.Collections.Generic.List[object]]::new()
$PreviousCpu = (Get-Process -Name python -ErrorAction SilentlyContinue | Measure-Object -Property CPU -Sum).Sum
$PreviousTime = [DateTime]::UtcNow
while (-not $Process.HasExited) {
    Start-Sleep -Seconds $SampleSeconds
    $Process.Refresh()
    if ($Process.HasExited) { break }
    $Now = [DateTime]::UtcNow
    $Cpu = (Get-Process -Name python -ErrorAction SilentlyContinue | Measure-Object -Property CPU -Sum).Sum
    $Wall = ($Now - $PreviousTime).TotalSeconds
    $Util = if ($Wall -gt 0) { 100.0 * ($Cpu - $PreviousCpu) / ($Wall * $LogicalProcessors) } else { 0.0 }
    if ($Util -ge 0 -and $Util -le 100) { $Samples.Add([pscustomobject]@{timestamp_utc=$Now.ToString('o');total_python_cpu_percent=$Util}) }
    $PreviousCpu, $PreviousTime = $Cpu, $Now
}
$Process.WaitForExit(); $Process.Refresh()
[ordered]@{schema_version=1;note='aggregate utilization of parent and Python workers';samples=$Samples;exit_code=$Process.ExitCode} |
    ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $CpuReport
if (-not (Test-Path $SweepReport)) {
    throw "Sweep report missing; inspect $StdoutLog and $StderrLog."
}
exit $Process.ExitCode
