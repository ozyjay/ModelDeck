[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$Q4Worker,

    [Parameter(Mandatory)]
    [string]$BF16Worker,

    [Parameter(Mandatory)]
    [string]$Q4Route,

    [Parameter(Mandatory)]
    [string]$BF16Route,

    [ValidateRange(1, 16)]
    [int]$SeedRepeats = 1,

    [ValidateRange(0, 100)]
    [int]$StabilityRuns = 4,

    [ValidateRange(8, 256)]
    [int]$MaxLength = 256,

    [ValidateRange(1, 48)]
    [int]$DenoisingSteps = 48,

    [string]$PromptsFile,

    [string]$JsonOutput = 'var/q4-quality-evaluation.json'
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

$Runtime = (Resolve-Path '.venv-rocm72-q4/bin/python' -ErrorAction SilentlyContinue).Path
if (-not $Runtime) {
    throw 'Q4 runtime missing. Create .venv-rocm72-q4 and install the Q4 requirements.'
}
$Checkpoint = '/mnt/work/models/modeldeck/diffusiongemma-26b-a4b-it-gptq-q4-g32'
if (-not (Test-Path (Join-Path $Checkpoint 'q4-manifest.json'))) {
    throw "Q4 checkpoint manifest missing: $Checkpoint/q4-manifest.json"
}

$Env:MODELDECK_ROCM72_Q4_PYTHON = $Runtime
& $Runtime -c "from modeldeck.workers.diffusiongemma_q4 import load_diffusiongemma_q4"
if ($LASTEXITCODE -ne 0) {
    throw 'The Q4 environment does not have the current ModelDeck package. Run: python -m pip install --no-deps -e .'
}

$Arguments = @(
    './scripts/evaluate_diffusiongemma_q4.py',
    '--q4-worker', $Q4Worker,
    '--bf16-worker', $BF16Worker,
    '--q4-route', $Q4Route,
    '--bf16-route', $BF16Route,
    '--seed-repeats', "$SeedRepeats",
    '--stability-runs', "$StabilityRuns",
    '--max-length', "$MaxLength",
    '--denoising-steps', "$DenoisingSteps",
    '--json-output', $JsonOutput,
    '--leave-worker', 'q4'
)
if ($PromptsFile) {
    $Arguments += @('--prompts-file', $PromptsFile)
}

& $Runtime @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Q4 evaluation failed. Inspect $JsonOutput for the individual release gates."
}
