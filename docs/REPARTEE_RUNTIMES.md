# Repartee model runtimes

ModelDeck recognises two user-configured Repartee roles. Neither role has a built-in model
profile or cloud fallback.

## Strong model

`repartee-strong` accepts an autoregressive profile created from the
`gpt-oss-llama-vulkan` template. The supported artefact is the official consolidated
MXFP4 GGUF or the complete legacy three-shard MXFP4 release from
`ggml-org/gpt-oss-120b-GGUF`. The consolidated artefact at revision
`8d158cefb5f175c6f8842bbd8f68eca54d951ab4` passed physical smoke, repeated lifecycle, a
standard five-request benchmark, and a 30-minute sustained run with 285 successful requests,
clean process exit, and measured whole-device GTT recovery on the target Framework Desktop.
The OpenAI Transformers snapshot is shown as the source model but is not offered as an AMD
runtime.

Provision the allowlisted executable with:

```powershell
pwsh -NoProfile -File scripts/setup/setup_llama_vulkan.ps1
```

The default `gpt-oss-llama-vulkan` template uses full Vulkan offload. The separate
`gpt-oss-llama-vulkan-cpu-moe` template uses the fixed `vulkan-cpu-moe` preset with 20 MoE
layers on the CPU. It remains hardware-verification-gated and is not the default because it
does not yet have physical qualification evidence on the target machine. Arbitrary llama.cpp
arguments are never accepted through the management API. Each template retains its own
runtime-template identity and compatibility evidence. The runtime does not advertise Token
Trail traces and strips reasoning-only fields before returning responses. The tested
full-offload fingerprint used llama.cpp revision `f08c4c0d`, Mesa RADV 26.1.4, and the Radeon
8060S.

## Speech model

`repartee-speech` accepts the config-less but exact
`kyutai/moshiko-pytorch-bf16` snapshot. Provision its separate environment with:

```powershell
pwsh -NoProfile -File scripts/setup/setup_moshiko_rocm72.ps1
```

The stable WebSocket endpoint is `ws://127.0.0.1:8600/v1/speech/conversations`. The first
client message is:

```json
{"type":"session.start","model":"repartee-speech","audio":{"encoding":"pcm_s16le","sample_rate_hz":24000,"channels":1}}
```

Subsequent binary client frames are PCM16 microphone audio. Server JSON events include
`session.ready`, `transcript.delta`, `transcript.final`, `response.started`,
`response.completed` and `error`; server binary frames are PCM16 response audio. Clients may
send `response.cancel` or `session.close`. A frame is limited to one second of audio, only one
session is allowed, the voice is fixed to Moshiko, and raw audio is never persisted.

## Verification gate

Creating either Worker does not publish it for a Demo. Start and smoke-test the Worker,
then assign it to a compatible Route in an Event. An Event using the tested-working policy
will not publish until matching evidence exists for the exact Model revision and runtime.
