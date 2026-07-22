# Security and privacy

All defaults bind to `127.0.0.1`. LAN exposure requires an explicit future decision and
threat review. Open Day mode forces downloads off.

The frontend cannot submit commands, executable paths, raw runtime arguments, environment
variables, tokens, arbitrary filesystem paths, Docker access, camera data, uploads, or
cloud endpoints. Worker IDs select prevalidated manifests. Subprocesses use argument
arrays without a shell.

Additional runtime template packages require a local PowerShell installation step and an
exact operator-approved SHA-256; the browser cannot install or trust them. ModelDeck keeps
the trust record separately, verifies every loaded file, and fails closed after tampering.
A template can select only a code-registered launch builder and its bounded setting names.
Adding executable logic remains a reviewed code change, not a JSON or API capability.

The optional repository `.env` is trusted operator configuration, is gitignored, and is
read only by the PowerShell launch path. Its parser accepts a fixed name allowlist, treats
values literally without expansion or command execution, does not print values, and fails
closed on unknown, duplicate, malformed, or unterminated entries. Existing process
environment variables take precedence. `.env` is not readable or writable through the
management API or operator console.

Visitor prompts and generated content are not stored or logged. Supervisor log capture is
bounded to the latest 500 records per worker and redacts prompt, output, authorisation,
API-key, and token-shaped fields before persisting JSON Lines files under
`var/log/workers`. The location can be changed with `MODELDECK_LOG_DIR`. Full diagnostic
capture is not implemented in this slice. SQLite holds configuration and compatibility
evidence, not content history.

SpeechShift translation text, recognition audio and transcripts, and generated waveform
bytes remain request-scoped in memory. Recognition audio reaches its isolated inference
child only through stdin; neither audio nor transcript content is written to disk or logged.
The workers retain only content-free timings, token or byte counts, outcomes and temperature
metrics. Cancellation ownership is in memory, keyed by a caller-supplied request ID, and is
released at completion. Speech synthesis fails closed when either required temperature
sensor is unavailable; no request may start above the code-owned 55 °C GPU or 75 °C CPU
limits, and active generation is cancelled at 80 °C GPU or 95 °C CPU.

The SceneChat worker accepts visitor images only as strict base64 JPEG or PNG data URLs.
It rejects network/file URLs, SVG, mismatched MIME and magic bytes, multiple images,
requests over 12 MiB, decoded images over 8 MiB, dimensions over 4096 pixels, and images
over 16 million pixels. Images are oriented, fully decoded in memory, converted to RGB,
and released after the request. Neither the model nor processor may fetch a URL.

SceneChat prompts must exactly match the versioned local contract. The hidden safety prompt
is moved to the model's system role and only the curated question remains in the user turn.
Visible image text is explicitly untrusted and cannot override the system rules. Responses
are schema- and policy-validated once, with no repair, retry, content persistence, cloud
fallback, or alternate model routing. Uvicorn access logging is disabled, and sanitised
errors do not echo request bodies, base64 data, prompts, responses, credentials, tracebacks,
or local snapshot paths.

Only `MODELDECK_SCENECHAT_API_KEY` is inherited as a SceneChat-specific worker setting; the
loopback development default is `local`. Operators should set a local secret for the event
without writing it to logs or compatibility evidence.
