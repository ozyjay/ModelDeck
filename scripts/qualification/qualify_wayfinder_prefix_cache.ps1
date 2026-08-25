[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string[]]$Workers,

    [ValidateRange(3, 10)]
    [int]$Repetitions = 5,

    [string]$Output
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '../..')

if (-not (Test-Path '.venv/bin/python')) {
    throw 'Run pwsh -NoProfile -File scripts/setup/setup.ps1 first.'
}

$Arguments = @(
    './scripts/qualification/qualify_wayfinder_prefix_cache.py',
    '--workers'
) + $Workers + @('--repetitions', $Repetitions)
if ($Output) {
    $Arguments += @('--output', $Output)
}

& .venv/bin/python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "WayFinder prefix-cache qualification failed with exit code $LASTEXITCODE."
}
