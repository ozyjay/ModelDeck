param(
    [string]$Endpoint = 'http://127.0.0.1:8670',
    [string]$Model = 'worker-8ed7591e',
    [string]$WhisperSnapshot = '/mnt/work/models/huggingface/hub/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d',
    [string]$Output = 'var/verification/qwen3-tts-four-voices-20260723.json',
    [string[]]$Voices = @('vivian', 'serena'),
    [string[]]$Languages = @('en', 'fr', 'de'),
    [ValidateRange(1, 10)]
    [int]$Repetitions = 3,
    [ValidateRange(0, 120)]
    [double]$CooldownSeconds = 0,
    [string]$SampleOutputDir = 'var/verification/qwen3-tts-four-voice-samples-20260723'
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
    --output $Output `
    --voices $Voices `
    --languages $Languages `
    --repetitions $Repetitions `
    --cooldown-seconds $CooldownSeconds `
    --sample-output-dir $SampleOutputDir
if ($LASTEXITCODE -ne 0) { throw 'Qwen3-TTS physical qualification failed.' }
