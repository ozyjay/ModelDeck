$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$Python = if ($Env:MODELDECK_PYTHON) { $Env:MODELDECK_PYTHON } else { 'python3.12' }
try { & $Python --version 2>$null | Out-Null } catch {
    $Python = Get-ChildItem "$HOME/.pyenv/versions/*/bin/python3.12" -ErrorAction SilentlyContinue |
        Sort-Object FullName | Select-Object -Last 1 -ExpandProperty FullName
    if (-not $Python) { throw 'Python 3.12 is required.' }
}
if (-not (Test-Path '.venv')) { & $Python -m venv .venv }
& .venv/bin/python -m pip install --upgrade pip
& .venv/bin/python -m pip install -e '.[dev]'
Write-Host "ModelDeck environment ready at $PWD/.venv"
