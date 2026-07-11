[CmdletBinding()]
param([switch]$ControlPlaneOnly)

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

if ($ControlPlaneOnly) {
    Write-Host 'ModelDeck control-plane environment is ready.'
} else {
    $Runtime = '.venv-rocm72'
    if (-not (Test-Path "$Runtime/bin/python")) {
        & $Python -m venv $Runtime
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the isolated ROCm 7.2 environment.' }
    }
    & "$Runtime/bin/python" -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'Could not update pip in the ROCm environment.' }
    & "$Runtime/bin/python" -m pip install -r runtime/requirements-rocm72.txt
    if ($LASTEXITCODE -ne 0) { throw 'Could not install the pinned ROCm runtime.' }
    & "$Runtime/bin/python" -m pip install --no-deps -e .
    if ($LASTEXITCODE -ne 0) { throw 'Could not install the ModelDeck worker into the ROCm environment.' }

    & "$Runtime/bin/python" -c "import PIL, torch, torchvision, transformers; from transformers import DiffusionGemmaForBlockDiffusion; from transformers.models.gemma4.processing_gemma4 import Gemma4Processor; print('torch', torch.__version__); print('hip', torch.version.hip); print('torchvision', torchvision.__version__); print('transformers', transformers.__version__); print('pillow', PIL.__version__); print('diffusiongemma_import', 'ok'); print('cuda_available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not visible in this session')"
    if ($LASTEXITCODE -ne 0) { throw 'The ROCm runtime import probe failed.' }
    Write-Host 'ModelDeck control-plane and ROCm environments are ready.'
}
