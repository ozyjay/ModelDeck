# Repartee model runtimes

ModelDeck recognises two user-configured Repartee roles. Neither role has a built-in model
profile or cloud fallback.

## Strong model

`repartee-strong` accepts an autoregressive profile created from the
`gpt-oss-llama-vulkan` template. The supported artefact is the three-shard MXFP4 release
from `ggml-org/gpt-oss-120b-GGUF`, pinned by HuggingFacePull at revision
`a48e247410f0cc543e586c661ce63d82e79f2ad6`. The OpenAI Transformers snapshot is shown as
the source model but is not offered as an AMD runtime.

Provision the allowlisted executable with:

```powershell
pwsh -NoProfile -File scripts/setup_llama_vulkan.ps1
```

The default preset uses full Vulkan offload. Compatibility testing may instead select the
fixed `vulkan-cpu-moe` preset; arbitrary llama.cpp arguments are never accepted through the
management API. The runtime does not advertise Token Trail traces and strips reasoning-only
fields before returning responses.

## Speech model

`repartee-speech` accepts the config-less but exact
`kyutai/moshiko-pytorch-bf16` snapshot. Provision its separate environment with:

```powershell
pwsh -NoProfile -File scripts/setup_moshiko_rocm72.ps1
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

Creating either runtime configuration does not select it for a demo. Start and smoke-test the
worker, complete lifecycle evidence, then select the compatible profile for its reserved alias.
The gateway filters profiles marked `hardware_verification_required` until a matching
`tested-working` fingerprint exists for the exact model revision and runtime.
