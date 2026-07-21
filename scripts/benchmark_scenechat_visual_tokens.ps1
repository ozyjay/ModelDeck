[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Worker280,
    [Parameter(Mandatory)][string]$Worker140,
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

$Arguments = @(
    'scripts/benchmark_scenechat_visual_tokens.py',
    '--worker-280', $Worker280,
    '--worker-140', $Worker140,
    '--warmups', $Warmups,
    '--runs', $Runs,
    '--maximum-temperature-celsius', $MaximumTemperatureCelsius,
    '--cooldown-temperature-celsius', $CooldownTemperatureCelsius,
    '--requests-per-thermal-batch', $RequestsPerThermalBatch
)
if ($HumanReview) { $Arguments += '--human-review' }
& .venv/bin/python @Arguments
if ($LASTEXITCODE -ne 0) { throw 'The SceneChat visual-token benchmark failed.' }
