"""Build the retrieval chunk dataset from processed pages."""

import argparse
from pathlib import Path

import pandas as pd

from aviation_rag.chunking import build_word_window_chunks


DEFAULT_MAX_WORDS = 140
DEFAULT_OVERLAP_WORDS = 25


def build_chunk_dataset(
    project_root: Path,
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> pd.DataFrame:
    """Build and save retrieval chunks."""

    pages_path = (
        project_root
        / "data"
        / "processed"
        / "pages.jsonl"
    )

    output_path = (
        project_root
        / "data"
        / "processed"
        / "chunks.jsonl"
    )

    if not pages_path.exists():
        raise FileNotFoundError(
            f"Page dataset not found: {pages_path}"
        )

    pages = pd.read_json(pages_path, lines=True)

    chunks = build_word_window_chunks(
        pages=pages,
        max_words=max_words,
        overlap_words=overlap_words,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks.to_json(
        output_path,
        orient="records",
        lines=True,
        force_ascii=False,
        date_format="iso",
    )

    print("Chunk dataset built successfully.")
    print(f"Chunks: {len(chunks)}")
    print(f"Maximum words: {chunks['word_count'].max()}")
    print(f"Output: {output_path}")

    return chunks


def main() -> None:
    """Run the chunk-dataset command-line interface."""

    parser = argparse.ArgumentParser(
        description="Build retrieval chunks from processed pages."
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )

    parser.add_argument(
        "--max-words",
        type=int,
        default=DEFAULT_MAX_WORDS,
    )

    parser.add_argument(
        "--overlap-words",
        type=int,
        default=DEFAULT_OVERLAP_WORDS,
    )

    args = parser.parse_args()

    build_chunk_dataset(
        project_root=args.project_root.resolve(),
        max_words=args.max_words,
        overlap_words=args.overlap_words,
    )


if __name__ == "__main__":
    main()