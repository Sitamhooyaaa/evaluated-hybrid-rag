import pandas as pd

CHUNK_METADATA_COLUMNS = [
    "document_id",
    "title",
    "document_type",
    "authority_level",
    "gazette_number",
    "publication_date",
    "effective_from",
    "effective_date_note",
    "reporting_period_start",
    "reporting_period_end",
    "source_url",
    "page_number",
]

REQUIRED_PAGE_COLUMNS = set(
    CHUNK_METADATA_COLUMNS
    + ["text", "retrieval_eligible"]
)

"""Chunking utilities for retrieval documents."""


def split_text_by_words(
    text: str,
    max_words: int = 140,
    overlap_words: int = 25,
) -> list[str]:
    """Split text into overlapping word windows."""

    if max_words <= 0:
        raise ValueError("max_words must be greater than zero")

    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError(
            "overlap_words must be at least zero and smaller than max_words"
        )

    words = text.split()

    if not words:
        return []

    step_size = max_words - overlap_words
    chunks = []

    for start in range(0, len(words), step_size):
        chunk_words = words[start : start + max_words]
        chunks.append(" ".join(chunk_words))

        if start + max_words >= len(words):
            break

    return chunks

def build_word_window_chunks(
    pages: pd.DataFrame,
    max_words: int = 140,
    overlap_words: int = 25,
) -> pd.DataFrame:
    """Build retrieval chunks from eligible page records."""

    missing_columns = REQUIRED_PAGE_COLUMNS - set(pages.columns)

    if missing_columns:
        raise ValueError(
            "Page dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    eligible_pages = pages.loc[
        pages["retrieval_eligible"]
    ]

    chunk_records = []

    for _, page in eligible_pages.iterrows():
        page_chunks = split_text_by_words(
            text=page["text"],
            max_words=max_words,
            overlap_words=overlap_words,
        )

        metadata = {
            column: page[column]
            for column in CHUNK_METADATA_COLUMNS
        }

        for chunk_index, chunk_text in enumerate(page_chunks):
            chunk_records.append(
                {
                    **metadata,
                    "chunk_id": (
                        f"{page['document_id']}"
                        f"::page_{int(page['page_number']):03d}"
                        f"::chunk_{chunk_index:02d}"
                    ),
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text,
                    "chunk_strategy": (
                        f"word_window_{max_words}"
                        f"_overlap_{overlap_words}"
                    ),
                    "word_count": len(chunk_text.split()),
                    "character_count": len(chunk_text),
                }
            )

    chunks = pd.DataFrame(chunk_records)

    if not chunks.empty and not chunks["chunk_id"].is_unique:
        raise ValueError("Generated chunk IDs are not unique")

    return chunks

