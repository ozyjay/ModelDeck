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
$Runtime = '.venv-whisper-rocm72'

Write-Host '[1/5] Preparing the isolated Whisper ROCm environment...'
if (-not (Test-Path "$Runtime/bin/python")) {
    & $Python -m venv $Runtime
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the isolated Whisper ROCm environment.' }
}
Write-Host '[2/5] Updating pip...'
& "$Runtime/bin/python" -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Could not update pip in the Whisper environment.' }
Write-Host '[3/5] Installing the pinned offline runtime...'
& "$Runtime/bin/python" -m pip install -r runtime/requirements-whisper-rocm72.txt
if ($LASTEXITCODE -ne 0) { throw 'Could not install the pinned Whisper ROCm runtime.' }
Write-Host '[4/5] Installing the ModelDeck Worker...'
& "$Runtime/bin/python" -m pip install --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw 'Could not install ModelDeck in the Whisper environment.' }
Write-Host '[5/5] Checking ROCm and Whisper imports...'
& "$Runtime/bin/python" -c "from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor; import torch; print('torch', torch.__version__); print('hip', torch.version.hip); print('transformers', __import__('transformers').__version__); print('cuda_available', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw 'The Whisper ROCm import probe failed.' }
Write-Host 'The isolated Whisper ROCm 7.2 runtime is ready.'
