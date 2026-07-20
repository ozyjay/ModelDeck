[CmdletBinding()]
param(
    [string]$ModelId = 'google/gemma-4-12B-it',
    [ValidateRange(65, 90)][double]$MaximumTemperatureCelsius = 80,
    [ValidateRange(45, 75)][double]$CooldownTemperatureCelsius = 65
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
if (-not (Test-Path '.venv/bin/python')) { throw 'Run scripts/setup.ps1 first.' }
if ($CooldownTemperatureCelsius -ge $MaximumTemperatureCelsius) {
    throw 'CooldownTemperatureCelsius must be below MaximumTemperatureCelsius.'
}

& .venv/bin/python scripts/probe_scenechat_visual_tokens.py `
    --model-id $ModelId `
    --maximum-temperature-celsius $MaximumTemperatureCelsius `
    --cooldown-temperature-celsius $CooldownTemperatureCelsius
if ($LASTEXITCODE -ne 0) { throw 'The guarded SceneChat visual-token probe failed.' }
