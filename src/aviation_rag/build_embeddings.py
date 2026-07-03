"""Build persisted semantic embeddings for retrieval chunks."""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aviation_rag.embedding_artifacts import (
    save_embedding_artifact,
)


def build_embedding_artifacts(
    project_root: Path,
    model_loader: Callable[[str], Any] | None = None,
) -> np.ndarray:
    """Encode chunks once and save validated embedding artifacts."""

    chunks_path = (
        project_root
        / "data"
        / "processed"
        / "chunks.jsonl"
    )
    config_path = (
        project_root
        / "evaluation"
        / "retrieval_config.json"
    )
    embeddings_path = (
        project_root
        / "artifacts"
        / "bge_chunk_embeddings.npy"
    )
    metadata_path = (
        project_root
        / "artifacts"
        / "bge_chunk_embeddings.json"
    )

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunk dataset not found: {chunks_path}"
        )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Retrieval configuration not found: {config_path}"
        )

    chunks = pd.read_json(
        chunks_path,
        lines=True,
    )

    config = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    model_name = config[
        "semantic_retriever"
    ]["model"]

    if model_loader is None:
        from sentence_transformers import (
            SentenceTransformer,
        )

        model_loader = SentenceTransformer

    model = model_loader(model_name)

    embeddings = model.encode_document(
        chunks["chunk_text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    save_embedding_artifact(
        embeddings=embeddings,
        chunks=chunks,
        model_name=model_name,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
    )

    print("Embedding artifacts built successfully.")
    print(f"Model: {model_name}")
    print(f"Matrix shape: {embeddings.shape}")
    print(f"Embeddings: {embeddings_path}")
    print(f"Metadata: {metadata_path}")

    return embeddings


def main() -> None:
    """Run the embedding artifact command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Build persisted semantic embeddings "
            "for retrieval chunks."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )

    args = parser.parse_args()

    build_embedding_artifacts(
        project_root=args.project_root.resolve(),
    )


if __name__ == "__main__":
    main()