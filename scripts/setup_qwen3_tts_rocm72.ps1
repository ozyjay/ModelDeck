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
$Runtime = '.venv-qwen3-tts-rocm72'

if (-not (Test-Path "$Runtime/bin/python")) {
    & $Python -m venv $Runtime
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the isolated Qwen3-TTS ROCm environment.' }
}
& "$Runtime/bin/python" -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Could not update pip in the Qwen3-TTS environment.' }
& "$Runtime/bin/python" -m pip install -r runtime/requirements-qwen3-tts-rocm72.txt
if ($LASTEXITCODE -ne 0) { throw 'Could not install the pinned Qwen3-TTS ROCm runtime.' }
& "$Runtime/bin/python" -m pip install --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw 'Could not install the ModelDeck Worker in the Qwen3-TTS environment.' }
& "$Runtime/bin/python" -c "from qwen_tts import Qwen3TTSModel; import torch, torchaudio, transformers; print('torch', torch.__version__); print('hip', torch.version.hip); print('torchaudio', torchaudio.__version__); print('transformers', transformers.__version__); print('qwen_tts_import', 'ok'); print('cuda_available', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw 'The Qwen3-TTS ROCm import probe failed.' }
Write-Host 'The isolated Qwen3-TTS ROCm 7.2 runtime is ready.'
