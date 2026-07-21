[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$Python = if ($Env:MODELDECK_PYTHON) {
    $Env:MODELDECK_PYTHON
} elseif (Test-Path '.venv/bin/python') {
    '.venv/bin/python'
} else {
    'python3.12'
}
$Runtime = '.venv-marian-cpu'

if (-not (Test-Path "$Runtime/bin/python")) {
    & $Python -m venv $Runtime
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the isolated Marian CPU environment.' }
}
& "$Runtime/bin/python" -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Could not update pip in the Marian CPU environment.' }
& "$Runtime/bin/python" -m pip install -r runtime/requirements-marian-cpu.txt
if ($LASTEXITCODE -ne 0) { throw 'Could not install the pinned Marian CPU runtime.' }
& "$Runtime/bin/python" -m pip install --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw 'Could not install the ModelDeck Worker in the Marian environment.' }
& "$Runtime/bin/python" -c "from transformers import MarianMTModel, MarianTokenizer; import torch; print('torch', torch.__version__); print('marian_import', 'ok')"
if ($LASTEXITCODE -ne 0) { throw 'The Marian CPU import probe failed.' }
Write-Host 'The isolated Marian CPU runtime is ready.'
