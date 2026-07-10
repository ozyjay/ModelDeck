$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
& .venv/bin/python -m ruff check backend tests
& .venv/bin/python -m pytest
