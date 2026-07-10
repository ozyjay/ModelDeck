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

& "$Runtime/bin/python" -c "import torch, transformers; print('torch', torch.__version__); print('hip', torch.version.hip); print('transformers', transformers.__version__); print('cuda_available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not visible in this session')"
if ($LASTEXITCODE -ne 0) { throw 'The ROCm runtime import probe failed.' }
Write-Host 'The isolated ROCm 7.2 worker environment is ready. Fedora stock packages were not changed.'
