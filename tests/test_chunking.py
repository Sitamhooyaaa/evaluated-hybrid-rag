import pytest
import pandas as pd

from aviation_rag.chunking import (
    build_word_window_chunks,
    split_text_by_words,
)


def test_short_text_produces_one_chunk() -> None:
    chunks = split_text_by_words(
        "one two three",
        max_words=5,
        overlap_words=1,
    )

    assert chunks == ["one two three"]


def test_long_text_produces_overlapping_chunks() -> None:
    chunks = split_text_by_words(
        "one two three four five six",
        max_words=4,
        overlap_words=2,
    )

    assert chunks == [
        "one two three four",
        "three four five six",
    ]


def test_empty_text_produces_no_chunks() -> None:
    assert split_text_by_words("   ") == []


@pytest.mark.parametrize(
    ("max_words", "overlap_words"),
    [
        (0, 0),
        (4, -1),
        (4, 4),
        (4, 5),
    ],
)
def test_invalid_chunk_parameters_raise_error(
    max_words: int,
    overlap_words: int,
) -> None:
    with pytest.raises(ValueError):
        split_text_by_words(
            "one two three",
            max_words=max_words,
            overlap_words=overlap_words,
        )

def make_page(**overrides) -> dict:
    """Create a complete page record for chunking tests."""

    page = {
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

    page.update(overrides)

    return page


def test_chunk_builder_preserves_metadata_and_overlap() -> None:
    pages = pd.DataFrame([make_page()])

    chunks = build_word_window_chunks(
        pages,
        max_words=4,
        overlap_words=2,
    )

    assert chunks["chunk_text"].tolist() == [
        "one two three four",
        "three four five six",
    ]

    assert chunks["chunk_id"].tolist() == [
        "document_a::page_002::chunk_00",
        "document_a::page_002::chunk_01",
    ]

    assert chunks["document_id"].tolist() == [
        "document_a",
        "document_a",
    ]

    assert chunks["page_number"].tolist() == [2, 2]
    assert chunks["source_url"].nunique() == 1


def test_chunk_builder_ignores_excluded_pages() -> None:
    pages = pd.DataFrame(
        [
            make_page(),
            make_page(
                document_id="document_b",
                retrieval_eligible=False,
            ),
        ]
    )

    chunks = build_word_window_chunks(pages)

    assert set(chunks["document_id"]) == {"document_a"}

def test_chunk_builder_rejects_missing_columns() -> None:
    pages = pd.DataFrame([make_page()]).drop(
        columns=["source_url"]
    )

    with pytest.raises(
        ValueError,
        match="Page dataset is missing columns",
    ):
        build_word_window_chunks(pages)


def test_chunk_builder_rejects_duplicate_chunk_ids() -> None:
    duplicate_page = make_page()

    pages = pd.DataFrame(
        [
            duplicate_page,
            duplicate_page,
        ]
    )

    with pytest.raises(
        ValueError,
        match="Generated chunk IDs are not unique",
    ):
        build_word_window_chunks(pages)