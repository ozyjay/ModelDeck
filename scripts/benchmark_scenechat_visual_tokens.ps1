[CmdletBinding()]
param(
    [string]$Worker70,
    [string]$Worker140,
    [string]$Worker280,
    [ValidateRange(3, 5)][int]$Warmups = 4,
    [ValidateRange(50, 1000)][int]$Runs = 50,
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
    '--runs', $Runs,
    '--maximum-temperature-celsius', $MaximumTemperatureCelsius,
    '--cooldown-temperature-celsius', $CooldownTemperatureCelsius,
    '--requests-per-thermal-batch', $RequestsPerThermalBatch
)
if ($HumanReview) { $Arguments += '--human-review' }
& .venv/bin/python @Arguments
if ($LASTEXITCODE -ne 0) { throw 'The SceneChat visual-token benchmark failed.' }
