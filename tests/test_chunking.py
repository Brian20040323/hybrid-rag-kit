import pytest

from hybrid_rag.chunking import chunk_text


def test_chunk_boundaries_keep_requested_overlap() -> None:
    chunks = chunk_text("doc", "abcdefghij", chunk_size=6, overlap=2)

    assert [(chunk.text, chunk.start, chunk.end) for chunk in chunks] == [
        ("abcdef", 0, 6),
        ("efghij", 4, 10),
    ]


@pytest.mark.parametrize("overlap", [-1, 6])
def test_chunking_rejects_invalid_overlap(overlap: int) -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("doc", "abcdefghij", chunk_size=6, overlap=overlap)
