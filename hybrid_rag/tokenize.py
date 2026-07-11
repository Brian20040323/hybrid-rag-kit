"""Simple CJK-aware tokenizer."""

from __future__ import annotations

import re

_WORD = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    text = (text or "").lower()
    return _WORD.findall(text)
