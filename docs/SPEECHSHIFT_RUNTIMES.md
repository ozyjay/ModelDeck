# SpeechShift runtimes

SpeechShift uses three exact, locally cached Hugging Face snapshots:

- `Helsinki-NLP/opus-mt-en-fr@dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18`;
- `Helsinki-NLP/opus-mt-en-de@6183067f769a302e3861815543b9f312c71b0ca4`; and
- `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice@85e237c12c027371202489a0ec509ded67b5e4b5`.

ModelDeck validates the snapshot inventory, revision and architecture before loading. Both
workers set local-files-only mode and never download weights or execute repository code.

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

## Thermal safety and qualification

Qwen3-TTS requires readable AMD GPU-edge and CPU-package sensors. It fails closed if either
is absent. New work is rejected above 55 °C GPU or 75 °C CPU; active generation is cancelled
at 80 °C GPU or 95 °C CPU. These thresholds are code-owned and cannot be changed through the
browser or API.

Normal verification is GPU-free. Before describing a physical Worker as tested working,
record objective evidence for the exact fingerprint:

1. prove the pinned snapshot and runtime versions;
2. load and warm the Worker with outbound network access unavailable;
3. translate representative English input in each direction-specific Worker;
4. synthesise each allowlisted voice and language, validating a mono 24 kHz WAV;
5. exercise duplicate IDs, busy rejection, cancellation, deadlines and stopped-primary
   fallback;
6. exercise start rejection and active cancellation at the thermal boundaries; and
7. stop the process, confirm GPU memory recovery, then repeat a clean start and request.

Physical qualification is intentionally separate from `scripts/verify.ps1` because it needs
the target GPU, installed isolated environments, cached weights, sensors, time and memory.
