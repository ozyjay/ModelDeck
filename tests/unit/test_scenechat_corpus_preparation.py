import importlib.util
import json
from pathlib import Path

import pytest
from modeldeck.scenechat_experiment import CATEGORIES, Corpus
from PIL import Image


def test_candidate_cannot_run_until_individual_image_and_label_approval(tmp_path):
    path = Path(__file__).resolve().parents[2] / "scripts/benchmarks/prepare_scenechat_corpus.py"
    spec = importlib.util.spec_from_file_location("corpus_preparation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sources, planned = {}, []
    for number in range(28):
        name = f"test-{number}"
        source = tmp_path / f"{name}.png"
        Image.new("RGB", (1280, 720), (number * 8, 0, 0)).save(source)
        sources[name] = str(source)
        planned.append(
            {
                "id": name,
                "category": CATEGORIES[number // 4],
                "split": "development" if number % 4 < 2 else "holdout",
                "prompt": "Test fixture",
                "draft_facts": {f"q{q + 1}": ["<script> is literal text, not markup"] for q in range(7)},
            }
        )
    plan_path, sources_path = tmp_path / "plan.json", tmp_path / "sources.json"
    plan_path.write_text(json.dumps({"images": planned}))
    sources_path.write_text(json.dumps(sources))
    output = tmp_path / "corpus"
    module.prepare(plan_path, sources_path, output)
    candidate = output / "corpus.candidate.json"
    with pytest.raises(ValueError):
        Corpus.model_validate_json(candidate.read_text())
    assert "<script>" not in (output / "review.html").read_text()
    review_path = output / "review-template.json"
    with pytest.raises(ValueError, match="approval"):
        module.approve(candidate, review_path, output / "corpus.json")
    review = json.loads(review_path.read_text())
    for item in review["images"]:
        item["approved"] = True
        item["approved_by"] = "test reviewer"
    review_path.write_text(json.dumps(review))
    module.approve(candidate, review_path, output / "corpus.json")
    approved = Corpus.model_validate_json((output / "corpus.json").read_text())
    approved.verify_files(output)
    assert len(approved.images) == 28
    assert all(Image.open(output / image.path).format == "JPEG" for image in approved.images)
    for item in approved.images:
        if item.split == "holdout":
            (output / item.path).unlink()
    # Development validation works even when holdout pixels are inaccessible.
    approved.verify_files(output, {i.id for i in approved.images if i.split == "development"})
    (output / approved.images[0].path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash changed"):
        approved.verify_files(output)
