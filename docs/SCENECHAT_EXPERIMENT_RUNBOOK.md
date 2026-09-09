# SceneChat configuration experiments

Implements the staged tooling in the [configuration proposal](SCENECHAT_CONFIGURATION_PROPOSAL_2026-09-09.md).
No configuration is qualified until holdout, sustained-load, human-review and application
checks pass. The tooling never publishes a route automatically.

## Prepare without inference

Use PowerShell from ModelDeck. Control/tests use `.venv`; freezing and direct inference
use `.venv-rocm72`. No command downloads models or installs dependencies.

```powershell
pwsh -NoProfile -File scripts/benchmarks/scenechat_experiment_tools.ps1 `
    -Action schemas --output var/scenechat-preparation/schemas
pwsh -NoProfile -File scripts/benchmarks/scenechat_experiment_tools.ps1 `
    -Action prepare-workers --output var/scenechat-preparation/worker-requests
pwsh -NoProfile -File scripts/benchmarks/scenechat_experiment_tools.ps1 -Action freeze-installation
```

Outputs are exclusive: existing evidence is not overwritten. The four Worker request
bodies pin Gemma E2B, Qwen 4B, 2B and 0.8B, using dedicated SceneChat templates, BF16,
8,192 context, 1,024 output and 140 visual tokens. Submit reviewed requests through the
existing management Worker creation flow. Do not publish them or reuse general-chat
Workers. The matrix validator checks the resolved 60-second deadline and other settings.

Before creating definitions/listeners, reconcile `ss -ltn`, stored Worker allocations and
the personal reservation table. Existing allocations extend beyond 8610–8624. This feature
assigns no persistent default port: document selected configurable loopback ports and record
them in the maintenance receipt. Allocated, reserved or occupied gateway ports fail.

Freezing copies ModelDeck's package into `.modeldeck/scenechat-experiments/bundles/<digest>`
and hashes the interpreter/dependency files. The supervisor verifies the receipt and uses
a code-owned frozen package path. Dependency drift blocks launch; this does not copy the
entire Python environment. Archive the environment/build artefacts before later upgrades
so rollback can restore dependencies as well as source. No web input controls launch paths.

The corpus schema requires 28 distinct 1280 × 720 JPEGs, provenance, non-visitor approval
and factual labels for `q1`–`q7`. Categories are `sparse`, `cluttered`, `workshop`,
`synthetic_people`, `outdoor`, `degraded` and `visible_text`: two development and two holdout
images per category. Paths are relative to the manifest and cannot escape its directory.
Human approval must reflect review of the actual pixels, not merely a generation prompt.
The older `var/benchmarks/scenechat*` reports all reference SceneChat's single synthetic
booth PNG; they do not contain the proposed varied corpus. Keep that PNG as a legacy anchor.

Export the stopped Worker API records into an object keyed `gemma-e2b`, `qwen-4b`, `qwen-2b`
and `qwen-08b`, then build the matrix and freeze a schedule:

```powershell
pwsh -NoProfile -File scripts/benchmarks/scenechat_experiment_tools.ps1 `
    -Action matrix --workers var/scenechat-preparation/workers.json `
    --bundle <bundle-digest> --output var/scenechat-preparation/matrix.json
pwsh -NoProfile -File scripts/benchmarks/benchmark_scenechat_visual_tokens.ps1 `
    -ConfigurationMatrix var/scenechat-preparation/matrix.json `
    -Corpus <corpus-directory>/corpus.json -Stage screen `
    -Schedule var/scenechat-preparation/screen.json
```

Matrix creation hashes every file in the exact model/processor snapshot. Execution checks
the inventory, Worker definition, bundle, corpus, questions, contract, policy and schedule.
Missing cache-class/dtype and dispatched-kernel observations remain unavailable; requested
attention settings are not kernel attestations.

## Register and execute in an approved window

```powershell
pwsh -NoProfile -File scripts/benchmarks/scenechat_experiment_tools.ps1 `
    -Action register --matrix var/scenechat-preparation/matrix.json `
    --schedule var/scenechat-preparation/screen.json
```

Registration requires stopped Workers and enables capture only for the stage's exact image
hashes. Use `--replace-stopped` for later stages; it archives the previous registration.
Development registrations exclude holdout hashes. Unapproved image bytes fail before
inference. Raw results, including contract-rejected results, are separate mode-0600 files
under `.modeldeck/scenechat-experiments/raw/<worker-uuid>`. Public errors and operational
logs remain content-free. Failures before an engine returns cannot provide raw output.

The maintenance JSON records `approved: true`, `approved_by`, timezone-aware `starts_at`
and `ends_at`, `allowed_worker_ids`, `stop_worker_ids`, `reserved_evaluation_ports` and
`load_mode` (`isolated` or `combined`). Sustained mode additionally records
`resident_workloads`, including the intended detector. Isolated runs refuse active Workers
outside the authorised stop set. Stopped services remain stopped for operator recovery;
there is no automatic restart after a safety abort.

```powershell
pwsh -NoProfile -File scripts/benchmarks/benchmark_scenechat_visual_tokens.ps1 `
    -ConfigurationMatrix var/scenechat-preparation/matrix.json `
    -Corpus <corpus-directory>/corpus.json -Stage screen `
    -Schedule var/scenechat-preparation/screen.json -EvaluationPort <reserved-port> `
    -MaintenanceReceipt <approved-window.json> -Execute
```

The isolated route has one Worker, no backup and a 60-second deadline. Journal events are
fsynced incrementally: an unfinished started attempt is interrupted; absent starts are
not run. Every execution has a new run directory. Request IDs reject stale Worker metrics.

Independent Tctl/GPU monitoring targets 0.45 seconds between samples. Missing sensors,
collection exceeding 0.5 seconds, ≥80°C or host availability below 16 GiB abort the run.
Start/cooldown requires ≤65°C. Records include RSS, GTT, host availability, swap and Worker
memory metrics. Review actual gaps and shutdown failures; unsafe evidence never passes.

## Stages and matched diagnostics

| Stage | Measured requests per configuration | Workload |
|---|---:|---|
| `screen` | 28 | Four development images × seven questions, Gemma first |
| `development` | 294 | 14 images × seven questions × three blocks |
| `holdout` | 490 | 14 unseen images × seven questions × five blocks; at most two finalists |
| `sustained` | 210 | One finalist under combined load, paced across at least two hours |

Each loaded block has two excluded warm-ups. Development/holdout order is fixed and
counterbalanced. Sustained mode keeps one configuration loaded; pacing is workload policy,
not inference latency. Record queueing and cooldown separately as availability evidence.

For a small matched diagnostic comparison, independently freeze screen schedules using
`-ExecutionPath direct`, `worker` and `gateway`. Direct mode requires one configuration and
launches the frozen ROCm package in an isolated process without an HTTP listener. It uses
the shared image decoder, curated model-side prompt and output validator. Worker mode
bypasses gateway forwarding. Do not pool diagnostic paths into qualification latency.
Direct mode requires the evaluation Worker stopped; stop other loads separately when
measuring isolated performance.

Measure timing overhead with separate matrices having `diagnostic_timing` false/true;
register the latter with `--diagnostic-timing`. Keep runs separate and compare identical
diagnostic screens. GPU synchronisation bounds fused inference timing; vision encoding,
prefill and first-token sub-stages remain unavailable. Further decoding/kernel tuning
requires new development experiments, not edits to frozen holdout configurations.

## Blind review and release decision

```powershell
pwsh -NoProfile -File scripts/benchmarks/scenechat_experiment_tools.ps1 `
    -Action export-review --schedule <schedule.json> --matrix <matrix.json> `
    --journal <run-directory>/attempts.jsonl --output <review-directory>
pwsh -NoProfile -File scripts/benchmarks/scenechat_experiment_tools.ps1 `
    -Action report --schedule <schedule.json> --journal <run-directory>/attempts.jsonl `
    --reviews <review-array.json> --review-directory <review-directory> --output <report.json>
pwsh -NoProfile -File scripts/benchmarks/scenechat_experiment_tools.ps1 `
    -Action finalists --report <development-report.json> --matrix <development-matrix.json> `
    --output <finalists.json>
```

Give reviewers only the blind output file, corpus/labels and review schema. Keep the private
mapping separate. Exact outputs deduplicate within image/question; every attempt keeps its
own validity. A fixed question-stratified sample of at least 20%, plus disputed/unsafe
outputs, needs a second distinct reviewer. Unresolved disputes fail. Conservative aggregation
prevents a second review hiding unsafe evidence. Empty output cannot pass factual coverage.

Reports separate all-attempt/success latency, failures, per-question/per-scene results and
descriptive binomial intervals. Repeated images are not independent visual tests. Finalists
rank passing arms by p95, median, then unsupported claims. No passing arm yields an explicit
`no_qualifier` decision. Stage reports alone never claim release qualification.

Complete the `PhysicalEvidence` schema with hashed observation files, stage report digests,
thermal/memory/recovery measurements, detector baseline and a minimum FPS fixed before
the trial, lifecycle drills and display timings for all seven questions. See SceneChat's
`docs/MODELDECK_CONFIGURATION_QUALIFICATION.md` for display receipts and hardware checks.

```powershell
pwsh -NoProfile -File scripts/benchmarks/scenechat_experiment_tools.ps1 `
    -Action release-decision --holdout <holdout-report.json> --sustained <sustained-report.json> `
    --physical-evidence <physical-evidence.json> --output <release-decision.json>
```

This checks gates/evidence hashes; it does not publish. Before publication approval, re-read
Open2026, archive its actual mapping and source/environment bundle, prepare a draft changing
only the `scenechat-vision` Worker UUID, and rehearse route/rollback in isolation. Preserve
`scene-analysis-v1` and no backup. After separate approval, use existing atomic publication
and verify serving identity. Rollback does not qualify the old 0.8B configuration. If no arm
passes, detector-only/replay remains the operational fallback.

The new synthetic candidate corpus is prepared at `var/corpora/scenechat-synthetic-v1/`.
Open `review.html` locally to inspect all 28 images and draft labels. The directory
includes originals, 1280 × 720 JPEGs, hashes, exact generation prompts and a
`review-template.json` with every approval initially false. Keep this preparation
gallery out of model development because it includes holdout. Synthetic results
are evidence for these scenes; they do not establish unrestricted camera accuracy.

Copy the review template to a review file, correct question-specific facts against
the pixels, and have the operator fill each image's `approved_by` and `approved`
fields. Import that actual review without editing the candidate receipt:

```powershell
pwsh -NoProfile -File scripts/benchmarks/prepare_scenechat_corpus.ps1 `
    -Candidate var/corpora/scenechat-synthetic-v1/corpus.candidate.json `
    -Review <completed-review.json> `
    -Output var/corpora/scenechat-synthetic-v1/corpus.approved.json
```

For a fresh candidate set, use `-Plan fixtures/scenechat-corpus/prompts.json
-Sources <id-to-generated-source-path.json> -Output <new-directory>` instead.
The importer preserves originals and records transforms; it never grants approval.

Run `pwsh -NoProfile -File scripts/verify.ps1`, benchmark-script Ruff checks and SceneChat's
offline suite. Physical comparison, measured timing overhead, human accuracy review,
sustained qualification and a winner-specific publication package remain pending until
their corpus, reviewers and maintenance prerequisites are ready.

Preparation verification on 10 September 2026: ModelDeck `verify.ps1` passed
790 tests (nine skipped), 25 frontend tests, lint, type checks and build checks.
Focused benchmark-script lint/format checks passed. SceneChat's documented offline
suite passed 175 tests with three physical tests deselected. No physical inference,
Worker lifecycle changes or routing publication were performed for this preparation.

`var/scenechat-preparation/qualification-status.json` records the current pending
decision and candidate digest. The same directory contains exported schemas and four
Worker creation request bodies; those bodies have not created or started Workers.
The frozen Worker bundle is
`.modeldeck/scenechat-experiments/bundles/3d6f4a91b92ac580b91db575798449d650869c71cf3f957e5293ddf08bc7ce36/`.
Its receipt hashes source, interpreter and installed dependency files. Dependencies
remain in the existing ROCm environment: verify them again before launch and archive
that environment separately when preparing rollback. Any code or dependency change
requires a new verified receipt before qualification.
