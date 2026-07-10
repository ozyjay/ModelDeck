$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
if (-not (Test-Path '.venv/bin/modeldeck-probe')) { throw 'Run scripts/setup.ps1 first.' }
& .venv/bin/modeldeck-probe @args

