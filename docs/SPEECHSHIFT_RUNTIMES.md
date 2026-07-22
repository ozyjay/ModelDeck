# SpeechShift runtimes

SpeechShift uses four exact, locally cached Hugging Face snapshots:

- `Helsinki-NLP/opus-mt-en-fr@dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18`;
- `Helsinki-NLP/opus-mt-en-de@6183067f769a302e3861815543b9f312c71b0ca4`; and
- `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice@85e237c12c027371202489a0ec509ded67b5e4b5`; and
- `openai/whisper-small.en@e8727524f962ee844a7319d92be39ac1bd25655a`.

ModelDeck validates the snapshot inventory, revision and architecture before loading. Both
workers set local-files-only mode and never download weights or execute repository code.

The pinned Whisper repository declares Apache-2.0 and provides safetensors weights. The
licence is compatible with local execution and redistribution subject to the Apache-2.0
notice conditions. The model card also warns that Whisper can make transcription errors
and may produce text that was not spoken; SpeechShift must treat its output as untrusted
visitor input, not an authoritative record. This review covers the model artefact, not the
licensing of any recorded audio.

## Runtime setup

Translation is deliberately isolated in a Float32 CPU environment:

```powershell
pwsh -NoProfile -File scripts/setup_marian_cpu.ps1
```

Qwen3-TTS uses its own ROCm 7.2 environment because its pinned dependencies differ from the
primary inference environment:

```powershell
pwsh -NoProfile -File scripts/setup_qwen3_tts_rocm72.ps1
```

Whisper uses a separate ROCm 7.2 environment so speech recognition remains independently
upgradable and auditable:

```powershell
pwsh -NoProfile -File scripts/setup_whisper_rocm72.ps1
```

The setup scripts prepare dependencies only. They do not create or start Workers, publish
Routes, fetch models or change system packages.

## Contracts

Translation Routes are direction-specific: `translation-en-fr-v1` and
`translation-en-de-v1`. Requests use `POST /v1/translations`, include a caller-generated
`request_id`, and must match the Route direction. Each Worker executes one request at a
time and rejects excess work rather than queueing visitor text.

Speech synthesis uses `speech-synthesis-v1` and `POST /v1/audio/speech`. It accepts only the
built-in voices `ryan` and `aiden`, languages `en`, `fr` and `de`, and WAV output. Successful
responses are mono 24 kHz PCM WAV. Arbitrary voice cloning, style instructions, paths,
fixtures, headers and runtime arguments are not accepted.

Both families support `POST /v1/requests/{request_id}/cancel` through the stable gateway.
Mocks are available for both translation directions and speech synthesis, and routed mock
traffic retains the `x-modeldeck-fallback: mock` label.

Speech recognition uses `speech-recognition-v1` at `POST /v1/audio/transcriptions`. The
published API Model ID is `speechshift-stt`. Its JSON request contains a caller-generated
`request_id`, `model`, `language: "en"`, `encoding: "pcm_s16le"`, `sample_rate_hz: 16000`,
`channels: 1`, and `audio_base64`. Decoded audio must contain one to 128,000 PCM16 samples
(eight seconds). The response contains `text`, English language metadata, and only duration
and elapsed-time metrics.

The parent Worker never loads Whisper weights. Each request starts an allowlisted child
process group, sends PCM bytes over stdin, reads the transcript over stdout, and then lets
the child exit. Cancellation, timeout, sensor failure and thermal cut-off terminate the
whole process group. Audio and transcript content are never written to disk or included in
Worker logs or metrics. Only one recognition request may be active.

## Thermal safety and qualification

Qwen3-TTS and Whisper require readable AMD GPU-edge and CPU-package sensors. They fail
closed if either is absent. New work is rejected above 55 °C GPU or 75 °C CPU; active work is cancelled
at 80 °C GPU or 95 °C CPU. These thresholds are code-owned and cannot be changed through the
browser or API.

Normal verification is GPU-free. Before describing a physical Worker as tested working,
record objective evidence for the exact fingerprint:

1. prove the pinned snapshot and runtime versions;
2. load and warm the Worker with outbound network access unavailable;
3. translate representative English input in each direction-specific Worker;
4. transcribe bounded mono 16 kHz PCM16 with Whisper and validate safe response metrics;
5. synthesise each allowlisted voice and language, validating a mono 24 kHz WAV;
6. exercise duplicate IDs, busy rejection, cancellation, deadlines and stopped-primary
   fallback;
7. exercise start rejection and active cancellation at the thermal boundaries; and
8. stop the process, confirm GPU memory recovery, then repeat a clean start and request.

Physical qualification is intentionally separate from `scripts/verify.ps1` because it needs
the target GPU, installed isolated environments, cached weights, sensors, time and memory.
Run it explicitly from the control environment after setting the isolated interpreter:

```powershell
$Env:MODELDECK_RUN_WHISPER_HARDWARE_TESTS = '1'
$Env:MODELDECK_WHISPER_PYTHON = '.venv-whisper-rocm72/bin/python'
.venv/bin/python -m pytest tests/hardware/test_speech_recognition_rocm.py -v
```
