"""Document chunking for long markdown / text corpora."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChunkConfig:
    """Configuration for text chunking."""

    chunk_size: int = 180
    overlap: int = 40


@dataclass
class Chunk:
    chunk_id: str
    source_id: str
    text: str
    start: int
    end: int


def chunk_text(
    source_id: str,
    text: str,
    chunk_size: int = 180,
    overlap: int = 40,
    config: ChunkConfig | None = None,
) -> list[Chunk]:
    if config is not None:
        chunk_size = config.chunk_size
        overlap = config.overlap
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [Chunk(f"{source_id}#0", source_id, text, 0, len(text))]

    chunks: list[Chunk] = []
    i = 0
    idx = 0
    step = max(chunk_size - overlap, 1)
    while i < len(text):
        piece = text[i : i + chunk_size]
        # prefer break on Chinese/English punctuation
        cut = max(piece.rfind("。"), piece.rfind("！"), piece.rfind("？"), piece.rfind("\n"), piece.rfind(". "))
        if cut >= chunk_size // 3:
            piece = piece[: cut + 1]
        end = i + len(piece)
        chunks.append(Chunk(f"{source_id}#{idx}", source_id, piece.strip(), i, end))
        idx += 1
        if end >= len(text):
            break
        i += max(len(piece) - overlap, 1)
    return chunks
