"""Import built-in generated images without treating draft labels as human approval."""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from pathlib import Path

from modeldeck.scenechat_experiment import Corpus, digest, file_digest
from modeldeck.scenechat_experiment_cli import write_new
from PIL import Image


def prepare(plan_path: Path, sources_path: Path, output: Path) -> None:
    plan = json.loads(plan_path.read_text())
    sources = json.loads(sources_path.read_text())
    ids = {image["id"] for image in plan["images"]}
    if len(plan["images"]) != 28 or len(ids) != 28 or set(sources) != ids:
        raise ValueError("Exactly one generated source for each of the 28 planned images is required")
    if any(not re.fullmatch(r"[a-z0-9_-]+", name) for name in ids):
        raise ValueError("Candidate image IDs must be safe local filenames")
    # Validate every source before creating the candidate directory.
    for name in sources.values():
        with Image.open(name) as image:
            if abs(image.width / image.height - 16 / 9) > 0.02:
                raise ValueError("Generated images must retain the requested 16:9 framing")
            image.verify()
    output.mkdir(parents=True, exist_ok=False)
    (output / "originals").mkdir()
    (output / "images").mkdir()
    images = []
    cards = []
    for item in plan["images"]:
        source = Path(sources[item["id"]])
        original = output / "originals" / f"{item['id']}{source.suffix}"
        shutil.copyfile(source, original)
        destination = output / "images" / f"{item['id']}.jpg"
        with Image.open(source) as image:
            dimensions = list(image.size)
            resized = image.convert("RGB").resize((1280, 720), Image.Resampling.LANCZOS)
            resized.save(destination, "JPEG", quality=95, subsampling=0)
        images.append(
            {
                "version": 1,
                "id": item["id"],
                "path": str(destination.relative_to(output)),
                "sha256": file_digest(destination),
                "category": item["category"],
                "split": item["split"],
                "provenance": "Synthetic; built-in image_gen; text-only generation, no visitor input.",
                "non_visitor": True,
                "approved_by": None,
                "draft_facts": item["draft_facts"],
                "prompt_sha256": digest(item["prompt"]),
                "source_sha256": file_digest(original),
                "source_path": str(original.relative_to(output)),
                "source_dimensions": dimensions,
                "transform": "RGB; Lanczos resize to 1280x720; JPEG quality 95, subsampling 0; no crop",
            }
        )
        labels = "".join(
            f"<dt>{html.escape(q)}</dt><dd>{html.escape('; '.join(facts))}</dd>"
            for q, facts in item["draft_facts"].items()
        )
        cards.append(
            f"<article><h2>{html.escape(item['id'])}</h2>"
            f"<p>{html.escape(item['category'])} · {html.escape(item['split'])}</p>"
            f'<img loading="lazy" src="images/{item["id"]}.jpg" alt="Synthetic review candidate">'
            "<details><summary>Draft labels — verify against pixels</summary>"
            f"<dl>{labels}</dl></details></article>"
        )
    candidate = {
        "version": 1,
        "format": "modeldeck-scenechat-corpus-candidate",
        "approval_status": "pending-human-review",
        "generation_plan_sha256": digest(plan),
        "images": images,
    }
    write_new(output / "corpus.candidate.json", candidate)
    write_new(output / "prompts.json", plan)
    write_new(
        output / "review-template.json",
        {
            "version": 1,
            "candidate_sha256": digest(candidate),
            "images": [
                {
                    "id": i["id"],
                    "sha256": i["sha256"],
                    "approved_by": None,
                    "facts": i["draft_facts"],
                    "approved": False,
                }
                for i in images
            ],
        },
    )
    page = (
        '<!doctype html><html lang="en-AU"><meta charset="utf-8"><title>SceneChat candidate corpus</title>'
        "<style>body{font:16px system-ui;background:#f4f4f4;color:#222;margin:2rem}"
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:1rem}"
        "article{background:white;padding:1rem;border-radius:8px}img{width:100%}dd{margin-bottom:.7rem}</style>"
        "<h1>SceneChat synthetic candidate corpus</h1><p>Not approved. Review all images and factual labels; "
        "generation prompts are not ground truth. This gallery includes both development and holdout "
        "for corpus preparation; keep holdout away from model tuning.</p><main>"
        + "".join(cards)
        + "</main></html>"
    )
    (output / "review.html").write_text(page)


def approve(candidate_path: Path, review_path: Path, output: Path) -> None:
    candidate = json.loads(candidate_path.read_text())
    review = json.loads(review_path.read_text())
    if review["candidate_sha256"] != digest(candidate):
        raise ValueError("The reviewed candidate changed")
    reviewed = {r["id"]: r for r in review["images"]}
    if len(reviewed) != 28 or len(review["images"]) != 28:
        raise ValueError("Every image requires an individual approval")
    images = []
    for item in candidate["images"]:
        decision = reviewed[item["id"]]
        if (
            decision["approved"] is not True
            or not decision["approved_by"]
            or decision["sha256"] != item["sha256"]
        ):
            raise ValueError(f"Image/label approval missing or stale: {item['id']}")
        images.append(
            {
                k: item[k]
                for k in ("version", "id", "path", "sha256", "category", "split", "provenance", "non_visitor")
            }
            | {"approved_by": decision["approved_by"], "facts": decision["facts"]}
        )
    corpus = Corpus(images=images)
    corpus.verify_files(candidate_path.parent)
    if output.parent.resolve() != candidate_path.parent.resolve():
        raise ValueError("Keep the approved manifest beside its relative image paths")
    write_new(output, corpus.model_dump())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.candidate and args.review and not args.plan and not args.sources:
        approve(args.candidate, args.review, args.output)
    elif args.plan and args.sources and not args.candidate and not args.review:
        prepare(args.plan, args.sources, args.output)
    else:
        parser.error("Supply plan/sources for preparation, or candidate/review for approved manifest import")


if __name__ == "__main__":
    main()
