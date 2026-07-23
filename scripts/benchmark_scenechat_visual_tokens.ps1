[CmdletBinding()]
param(
    [string]$Worker70,
    [string]$Worker140,
    [string]$Worker280,
    [ValidateRange(2, 5)][int]$Warmups = 2,
    [Alias('Runs')][ValidateRange(10, 1000)][int]$RunsPerQuestion = 10,
    [ValidateSet('isolated', 'combined')][string]$LoadMode = 'isolated',
    [ValidateRange(0, 86400)][int]$MinimumDurationSeconds = 0,
    [switch]$HumanReview,
    [ValidateRange(65, 90)][double]$MaximumTemperatureCelsius = 80,
    [ValidateRange(45, 75)][double]$CooldownTemperatureCelsius = 65,
    [ValidateRange(1, 10)][int]$RequestsPerThermalBatch = 2
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
if (-not (Test-Path '.venv/bin/python')) { throw 'Run scripts/setup.ps1 first.' }
if ($CooldownTemperatureCelsius -ge $MaximumTemperatureCelsius) {
    throw 'CooldownTemperatureCelsius must be below MaximumTemperatureCelsius.'
}
if ($MinimumDurationSeconds -gt 0 -and $LoadMode -ne 'combined') {
    throw 'MinimumDurationSeconds requires LoadMode combined.'
}

$Workers = @(
    if ($Worker70) { @('--worker-70', $Worker70) }
    if ($Worker140) { @('--worker-140', $Worker140) }
    if ($Worker280) { @('--worker-280', $Worker280) }
)
if ($Workers.Count -eq 0) {
    throw 'Supply at least one of Worker70, Worker140 or Worker280.'
}

$Arguments = @(
    'scripts/benchmark_scenechat_visual_tokens.py',
    $Workers,
    '--warmups', $Warmups,
    '--runs-per-question', $RunsPerQuestion,
    '--load-mode', $LoadMode,
    '--minimum-duration-seconds', $MinimumDurationSeconds,
    '--maximum-temperature-celsius', $MaximumTemperatureCelsius,
    '--cooldown-temperature-celsius', $CooldownTemperatureCelsius,
    '--requests-per-thermal-batch', $RequestsPerThermalBatch
)
if ($HumanReview) { $Arguments += '--human-review' }
& .venv/bin/python @Arguments
if ($LASTEXITCODE -ne 0) { throw 'The SceneChat visual-token benchmark failed.' }
