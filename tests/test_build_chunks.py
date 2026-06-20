from pathlib import Path

import pandas as pd
import pytest

from aviation_rag.build_chunks import build_chunk_dataset


def make_page() -> dict:
    """Create a complete page record for pipeline testing."""

    return {
        "document_id": "document_a",
        "title": "Test Document",
        "document_type": "gazette",
        "authority_level": "official",
        "gazette_number": "P.U. TEST",
        "publication_date": "2024-01-01",
        "effective_from": "2024-02-01",
        "effective_date_note": "Test note",
        "reporting_period_start": None,
        "reporting_period_end": None,
        "source_url": "https://example.com/document.pdf",
        "page_number": 2,
        "text": "one two three four five six",
        "retrieval_eligible": True,
    }


def test_build_chunk_dataset_saves_jsonl(
    tmp_path: Path,
) -> None:
    processed_directory = (
        tmp_path / "data" / "processed"
    )

    processed_directory.mkdir(parents=True)

    pages_path = processed_directory / "pages.jsonl"

    pd.DataFrame([make_page()]).to_json(
        pages_path,
        orient="records",
        lines=True,
    )

    chunks = build_chunk_dataset(
        project_root=tmp_path,
        max_words=4,
        overlap_words=2,
    )

    output_path = processed_directory / "chunks.jsonl"

    assert output_path.exists()
    assert len(chunks) == 2

    saved_chunks = pd.read_json(
        output_path,
        lines=True,
    )

    assert len(saved_chunks) == 2
    assert saved_chunks["chunk_id"].is_unique
    assert saved_chunks["word_count"].max() == 4


def test_build_chunk_dataset_requires_pages_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="Page dataset not found",
    ):
        build_chunk_dataset(project_root=tmp_path)