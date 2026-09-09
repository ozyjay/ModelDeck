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

Use the corpus preparation commands in
`docs/SCENECHAT_EXPERIMENT_RUNBOOK.md` to import individual human approvals.
Keep the historic booth illustration separate.
