"""Tokenization with optional jieba for Chinese."""

from __future__ import annotations

import re

_WORD = re.compile(r"[A-Za-z0-9_]+|[一-鿿]")

try:
    import jieba

    _JIEBA_AVAILABLE = True
except ImportError:
    _JIEBA_AVAILABLE = False


def tokenize(text: str) -> list[str]:
    """Tokenize text. Uses jieba for Chinese if available, regex fallback otherwise."""
    if not text or not text.strip():
        return []
    text_lower = text.lower()
    if _JIEBA_AVAILABLE:
        tokens: list[str] = []
        # jieba cuts CJK; regex picks up ASCII words
        for segment in jieba.cut(text_lower):
            segment = segment.strip()
            if not segment:
                continue
            if re.search(r"[一-鿿]", segment):
                tokens.append(segment)
            else:
                tokens.extend(_WORD.findall(segment))
        return [t for t in tokens if t]
    return _WORD.findall(text_lower)
