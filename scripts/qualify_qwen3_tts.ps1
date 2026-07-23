param(
    [string]$Endpoint = 'http://127.0.0.1:8669',
    [string]$Model = 'worker-7f93fdbf',
    [string]$WhisperSnapshot = '/mnt/work/models/huggingface/hub/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d',
    [string]$Output = 'var/verification/qwen3-tts-sampled-256-20260723.json'
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
if (-not (Test-Path '.venv-qwen3-tts-rocm72/bin/python')) {
    throw 'Run pwsh -NoProfile -File scripts/setup_qwen3_tts_rocm72.ps1 first.'
}
& .venv-qwen3-tts-rocm72/bin/python scripts/qualify_qwen3_tts.py `
    --endpoint $Endpoint `
    --model $Model `
    --whisper-snapshot $WhisperSnapshot `
    --output $Output
if ($LASTEXITCODE -ne 0) { throw 'Qwen3-TTS physical qualification failed.' }
