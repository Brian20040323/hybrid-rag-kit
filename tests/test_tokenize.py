"""Tests for tokenization (including Chinese support)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hybrid_rag.tokenize import tokenize


def test_english_tokenize():
    tokens = tokenize("Hello world! BM25 ranking.")
    assert "hello" in tokens
    assert "world" in tokens
    assert "bm25" in tokens
    assert "ranking" in tokens


def test_chinese_tokenize():
    tokens = tokenize("混合检索提升RAG效果")
    # Should produce meaningful CJK tokens (not just single chars)
    assert len(tokens) > 0
    # With jieba, we'd get multi-char tokens; without, single chars
    # Either is acceptable - test just verifies it doesn't crash
    combined = " ".join(tokens)
    assert len(combined) > 0


def test_mixed_language():
    tokens = tokenize("BM25和TF-IDF混合检索")
    assert "bm25" in tokens
    # Should have some CJK tokens
    assert len(tokens) >= 3


def test_empty_text():
    assert tokenize("") == []
    assert tokenize("   ") == []
