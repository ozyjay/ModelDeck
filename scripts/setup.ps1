$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$Candidates = @()
if ($Env:MODELDECK_PYTHON) { $Candidates += $Env:MODELDECK_PYTHON }
$Candidates += 'python3.12'
$Candidates += @(Get-ChildItem "$HOME/.pyenv/versions/*/bin/python3.12" -ErrorAction SilentlyContinue |
    Sort-Object FullName -Descending | Select-Object -ExpandProperty FullName)
$Python = $null
foreach ($Candidate in $Candidates) {
    try {
        $Version = & $Candidate --version 2>&1
        if ($LASTEXITCODE -eq 0 -and "$Version" -match '^Python 3\.12\.') { $Python = $Candidate; break }
    } catch { continue }
}
if (-not $Python) { throw 'Python 3.12 is required.' }
if (-not (Test-Path '.venv')) {
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the project virtual environment.' }
}
& .venv/bin/python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Could not update pip in the project virtual environment.' }
& .venv/bin/python -m pip install -e '.[dev]'
if ($LASTEXITCODE -ne 0) { throw 'Could not install ModelDeck in the project virtual environment.' }
Write-Host "ModelDeck environment ready at $PWD/.venv"
