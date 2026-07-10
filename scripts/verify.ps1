$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
if (-not (Test-Path '.venv/bin/python')) { throw 'Run scripts/setup.ps1 first.' }
& .venv/bin/python -m ruff check backend tests
if ($LASTEXITCODE -ne 0) { throw 'Ruff checks failed.' }
& .venv/bin/python -m ruff format --check backend tests
if ($LASTEXITCODE -ne 0) { throw 'Ruff formatting check failed.' }
& .venv/bin/python -m pytest
if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
