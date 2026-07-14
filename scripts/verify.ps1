$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
if (-not (Test-Path '.venv/bin/python')) { throw 'Run scripts/setup.ps1 first.' }
& .venv/bin/python -m ruff check backend tests
if ($LASTEXITCODE -ne 0) { throw 'Ruff checks failed.' }
& .venv/bin/python -m ruff format --check backend tests
if ($LASTEXITCODE -ne 0) { throw 'Ruff formatting check failed.' }
if (-not (Test-Path 'frontend/node_modules')) { throw 'Run pwsh -NoProfile -File scripts/setup.ps1 first.' }
& npm --prefix frontend run check
if ($LASTEXITCODE -ne 0) { throw 'Operator console type checks failed.' }
& npm --prefix frontend run test
if ($LASTEXITCODE -ne 0) { throw 'Operator console tests failed.' }
& (Join-Path $PSScriptRoot 'build_frontend.ps1') -Check
& .venv/bin/python -m pytest
if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
