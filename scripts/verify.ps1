$ErrorActionPreference = 'Stop'

& (Join-Path $PSScriptRoot 'verification/verify.ps1')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
