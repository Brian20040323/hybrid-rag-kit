from __future__ import annotations

import json
import sys

from hybrid_rag import Document, Hit
from hybrid_rag.cli import main
from hybrid_rag.diversity import expand_query, mmr_select


def test_cli_runs_sparse_without_dense_download(tmp_path, monkeypatch, capsys) -> None:
    (tmp_path / "one.md").write_text("hybrid retrieval uses rank fusion", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["hybrid-rag", str(tmp_path), "rank fusion", "--top-k", "1"],
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output["indexed"] == 1
    assert output["retrieval_mode"] == "sparse_rrf"
    assert output["dense_fallback_reason"] is None
    assert output["citations"][0]["doc_id"] == "one#0"


def test_expansion_and_mmr_diversify_ranked_hits() -> None:
    assert len(expand_query("RAG fusion")) == 2
    hits = [
        Hit("a", "rag retrieval", 1.0, 1.0, 0.0, 0.1),
        Hit("b", "rag retrieval duplicate", 0.9, 0.9, 0.0, 0.09),
        Hit("c", "fusion diversity", 0.8, 0.8, 0.0, 0.08),
    ]

    selected = mmr_select("rag fusion", hits, top_k=2, lambda_mult=0.5)

    assert selected[0].doc_id == "a"
    assert selected[1].doc_id == "c"


def test_document_dataclass_accepts_metadata() -> None:
    assert Document("id", "text", {"source": "unit"}).meta == {"source": "unit"}
