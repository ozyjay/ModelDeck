[CmdletBinding()]
param(
    [string]$ManagementUrl = 'http://127.0.0.1:3600',
    [string]$GatewayUrl = 'http://127.0.0.1:8600',
    [ValidateSet('TokenTrail', 'TextDiffusionDemo', 'SceneChat', 'SpeechShift')]
    [string[]]$Demo = @(),
    [string]$DiffusionModel = 'text-diffusion-lab-q4'
)

$ErrorActionPreference = 'Stop'
$Python = Join-Path $PSScriptRoot '../../.venv/bin/python'
if (-not (Test-Path $Python)) { throw 'Run scripts/setup/setup.ps1 first.' }
$Arguments = @('-m', 'modeldeck.demo_preflight', '--management-url', $ManagementUrl,
    '--gateway-url', $GatewayUrl, '--diffusion-model', $DiffusionModel)
foreach ($Name in $Demo) { $Arguments += @('--demo', $Name) }
& $Python @Arguments
exit $LASTEXITCODE
