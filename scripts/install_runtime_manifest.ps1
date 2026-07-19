[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Manifest,
    [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{64}$')][string]$Sha256,
    [string]$DataDir = '.modeldeck'
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

if (-not (Test-Path -LiteralPath '.venv/bin/python' -PathType Leaf)) {
    throw 'Run scripts/setup.ps1 first.'
}
if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) {
    throw "Runtime manifest does not exist: $Manifest"
}

& .venv/bin/python -m modeldeck.runtime_manifest_cli $Manifest --sha256 $Sha256 --data-dir $DataDir
if ($LASTEXITCODE -ne 0) { throw 'Runtime manifest installation failed.' }

Write-Host 'Restart ModelDeck to load the installed trusted runtime templates.'
