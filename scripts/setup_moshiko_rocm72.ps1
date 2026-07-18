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
$Runtime = '.venv-moshi-rocm72'

if (-not (Test-Path "$Runtime/bin/python")) {
    & $Python -m venv $Runtime
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the isolated Moshiko environment.' }
}
& "$Runtime/bin/python" -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Could not update pip in the Moshiko environment.' }
& "$Runtime/bin/python" -m pip install -r runtime/requirements-moshiko-rocm72.txt
if ($LASTEXITCODE -ne 0) { throw 'Could not install the pinned Moshiko ROCm runtime.' }
& "$Runtime/bin/python" -m pip install --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw 'Could not install the ModelDeck worker in the Moshiko environment.' }
& "$Runtime/bin/python" -c "import aiohttp, moshi, sphn, torch; print('torch', torch.__version__); print('hip', torch.version.hip); print('cuda_available', torch.cuda.is_available()); print('moshi_import', 'ok')"
if ($LASTEXITCODE -ne 0) { throw 'The Moshiko ROCm import probe failed.' }
Write-Host 'The isolated Moshiko ROCm 7.2 runtime is ready.'
