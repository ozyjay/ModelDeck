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

Speech synthesis uses `speech-synthesis-v1` and `POST /v1/audio/speech`. Its code-owned
allowlist contains exactly the built-in voices `ryan`, `aiden`, `vivian` and `serena`,
with languages `en`, `fr` and `de` and WAV output. Successful responses are mono 24 kHz
PCM WAV. Arbitrary speakers, voice cloning, reference audio, style instructions, paths,
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

## Qwen3-TTS deployed qualification

The sampled 256-token configuration was physically qualified on 23 July 2026 against the
Radeon 8060S before publishing Open2026 Event revision 32. The deployed Worker
`7f93fdbf-6a01-4320-94da-ca9440b51283` uses the pinned Qwen snapshot above, BF16, standard
PyTorch SDPA, `do_sample=True`, `subtalker_dosample=True`, a resident warmed model, a
256 codec-token limit and a 75-second generation deadline. Sampling, seed, temperature,
style instructions and arbitrary speakers remain absent from the request contract.

Fixed synthetic English, French and German requests completed in 45.643, 53.835 and
50.630 seconds respectively. They produced 5.04, 5.92 and 5.52 seconds of audio, for
real-time factors of 9.056, 9.094 and 9.172. Because the Worker returns a complete WAV,
first-audio latency equalled request time. No output contained clipped PCM16 samples.
Multilingual Whisper transcription produced word-error rates of 0.20, 0.333 and 0.333;
only aggregate scores were retained.

The comparison observed a 2,634.984 MB global device-memory baseline with the resident
Worker and a 3,021.004 MB peak during requests. Peak temperatures were 63.0 °C GPU edge
and 71.125 °C CPU package. A cancellation request was acknowledged in 7.010 ms. Qwen did
not reach its stopping criterion within the five-second grace period, so ModelDeck failed
the Worker closed and returned the structured `cancellation_unresponsive` error in
5.218 seconds. Stopping the process recovered global device memory to 0.059 MB, and a
clean restart and smoke request passed.

A deliberately long request exercised the immutable deadline. ModelDeck returned the
structured fail-closed response after 80.081 seconds, including the five-second
cancellation grace, leaving 9.919 seconds before SpeechShift's 90-second HTTP timeout.
The process again recovered to 0.059 MB and restarted cleanly. The published
`speechshift-voice` Route was then confirmed ready through gateway port 8600 and returned
a 24 kHz mono WAV in a gateway smoke request.

The privacy-safe aggregate report is generated with:

```powershell
pwsh -NoProfile -File scripts/qualify_qwen3_tts.ps1
```

The qualifier defaults to three repetitions for Vivian and Serena in each supported
language. It records per-voice/per-language aggregate timing, duration, real-time factor,
amplitude, clipping, multilingual Whisper word-error rate, device-memory and temperature
measurements. Repeated waveforms and transcription content remain in memory only and are
cleared after measurement. One fixed synthetic WAV per voice/language is retained under
`var/verification` for explicit pronunciation and intelligibility review. Input text,
transcripts and audio content are not included in reports, Worker logs or metrics.

### Four-voice candidate qualification

The immutable candidate Worker `8ed7591e-6e43-46c3-a72e-b10dd9edc5da` was created on
23 July 2026 with the exact stored voice metadata
`ryan,aiden,vivian,serena`. It retains the pinned model revision, BF16, PyTorch SDPA,
sampled talker and subtalker generation, the 256 codec-token limit and the 75-second
deadline. Its live capability response advertises exactly the same four voices.

Three fixed synthetic samples completed successfully for every Vivian and Serena
language pair. All 18 successful outputs validated as mono 24 kHz PCM16 WAV and none
contained a clipped sample. The aggregate medians were:

| Voice | Language | Request | Audio | RTF | Whisper WER |
| --- | --- | ---: | ---: | ---: | ---: |
| Vivian | English | 20.106 s | 5.52 s | 3.642 | 0.000 |
| Vivian | French | 55.241 s | 6.08 s | 9.086 | 0.417 |
| Vivian | German | 9.857 s | 5.84 s | 1.621 | 0.250 |
| Serena | English | 50.036 s | 5.52 s | 9.177 | 0.100 |
| Serena | French | 48.105 s | 5.36 s | 9.164 | 0.417 |
| Serena | German | 9.953 s | 6.16 s | 1.616 | 0.250 |

Sampling produced material latency variation: successful requests ranged from 8.789 to
64.603 seconds, still below the Worker deadline. Peak absolute normalised amplitude was
0.957 for Vivian and 0.582 for Serena; median RMS amplitude ranged from 0.080 to 0.092
for Vivian and 0.034 to 0.051 for Serena. The highest global device-memory observation
was 3,303.020 MB. Maximum observed temperatures were 65.0 °C GPU edge and 81.5 °C CPU
package.

The uninterrupted batch completed all Vivian and Serena English samples before the
thermal admission guard rejected the remaining six requests with
`thermal_cooldown_required`. No rejected request began inference. After recovery, Serena
French and German completed 3/3 with 15-second governed cooling intervals and no
additional admission retry. Ryan and Aiden each passed a fixed English regression smoke,
including WAV, clipping and multilingual Whisper checks.

Vivian cancellation was acknowledged in 3.772 ms. Qwen did not cooperate within the
five-second grace, so ModelDeck returned `cancellation_unresponsive` after 5.291 seconds.
A valid bounded Serena deadline probe returned the same structured fail-closed error
after 80.058 seconds, leaving 9.942 seconds before SpeechShift's 90-second timeout.
Stopping after each destructive probe recovered global device memory to 0.059 MB, and
the candidate then restarted cleanly with its four-voice capability response intact.
Qualification log inspection found no fixed input-text or WAV-content marker.

Aggregate evidence is retained in
`var/verification/qwen3-tts-four-voices-thermal-partial-20260723.json`,
`var/verification/qwen3-tts-serena-fr-de-20260723.json` and
`var/verification/qwen3-tts-ryan-aiden-regression-20260723.json`. The six fixed synthetic
review samples are under
`var/verification/qwen3-tts-four-voice-samples-20260723`.

Automated qualification is complete. Manual pronunciation and intelligibility review of
the six fixed samples remains required before the candidate can replace the published
Worker. Until that review is approved, the Event draft and published revision continue
to reference the original two-voice Worker.
