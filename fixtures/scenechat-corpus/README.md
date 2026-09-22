# SceneChat synthetic corpus preparation

`prompts.json` records the 28 prompts used with built-in image generation and draft
question-specific labels. It is a generation receipt, not an approved corpus.
The generated assets and local review gallery are in
`var/corpora/scenechat-synthetic-v1/`. Originals are preserved; evaluation images
are RGB 1280 × 720 JPEGs with recorded hashes and resize settings.

Review every image and correct the labels against its pixels. Prompts are not
ground truth. The split reserves two images per category for development and two
for holdout. Corpus preparation reviewers can inspect both; model development
must not use holdout. Synthetic-scene qualification describes this corpus and
does not establish accuracy on all real-world camera scenes.

For a visual editing workflow, open `var/corpora/scenechat-synthetic-v1/review-editor.html`
in a Chromium-family browser, then open `review-by-jase.json` in the page. Edit the seven
plain-language answer fields, enter your name, and approve each image after checking it.
Save the review JSON regularly. Where the browser supports direct file access, Save writes
to the opened file; otherwise it downloads a replacement JSON file which must be moved
beside the candidate manifest and images. The editor checks the candidate and image hashes
before opening a review. It does not submit data to a service or import an approved corpus.

The preparation script creates this editor for new candidate sets. To recreate the editor
for the current candidate without changing its images or review JSON, run:

```powershell
pwsh -NoProfile -File scripts/benchmarks/prepare_scenechat_corpus.ps1 `
    -Candidate var/corpora/scenechat-synthetic-v1/corpus.candidate.json `
    -Editor -Output var/corpora/scenechat-synthetic-v1
```

Use the corpus preparation commands in
`docs/SCENECHAT_EXPERIMENT_RUNBOOK.md` to import individual human approvals.
Keep the historic booth illustration separate.
