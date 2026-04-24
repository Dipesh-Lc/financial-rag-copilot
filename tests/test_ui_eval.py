import json
from pathlib import Path
from types import SimpleNamespace

from app.ui import gradio_app


def test_load_eval_results_prefers_processed_metadata(monkeypatch, tmp_path: Path):
    metadata_dir = tmp_path / "processed" / "metadata"
    eval_dir = tmp_path / "eval"
    metadata_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)

    older = eval_dir / "eval_results_similarity.json"
    older.write_text(json.dumps({"strategy": "similarity", "num_questions": 2, "summary": {}}), encoding="utf-8")

    latest = metadata_dir / "eval_results_multi_query.json"
    latest.write_text(json.dumps({"strategy": "multi_query", "num_questions": 3, "summary": {"ok": True}}), encoding="utf-8")

    monkeypatch.setattr(gradio_app, "CONFIG", SimpleNamespace(metadata_dir=metadata_dir, eval_dir=eval_dir))

    result = gradio_app.load_eval_results()

    assert result["file"] == "eval_results_multi_query.json"
    assert result["strategy"] == "multi_query"
    assert result["num_questions"] == 3
