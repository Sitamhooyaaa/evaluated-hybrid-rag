"""Build and validate persisted semantic embedding artifacts."""
from pathlib import Path

import numpy as np
import hashlib
import json
import pandas as pd


REQUIRED_HASH_COLUMNS = {
    "chunk_id",
    "chunk_text",
}


def compute_chunk_dataset_sha256(
    chunks: pd.DataFrame,
) -> str:
    """Hash chunk identities, text, and row order."""

    missing_columns = (
        REQUIRED_HASH_COLUMNS - set(chunks.columns)
    )

    if missing_columns:
        raise ValueError(
            "Chunk dataset is missing hashing columns: "
            f"{sorted(missing_columns)}"
        )

    if chunks.empty:
        raise ValueError("Chunk dataset cannot be empty")

    if chunks[["chunk_id", "chunk_text"]].isna().any().any():
        raise ValueError(
            "Chunk IDs and text cannot contain missing values"
        )

    records = chunks[
        ["chunk_id", "chunk_text"]
    ].to_dict(orient="records")

    canonical_bytes = json.dumps(
        records,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(canonical_bytes).hexdigest()

def _compute_file_sha256(file_path: Path) -> str:
    """Return the SHA-256 fingerprint of a file."""

    return hashlib.sha256(
        file_path.read_bytes()
    ).hexdigest()


def save_embedding_artifact(
    embeddings: np.ndarray,
    chunks: pd.DataFrame,
    model_name: str,
    embeddings_path: Path,
    metadata_path: Path,
) -> None:
    """Save normalized chunk embeddings and their metadata."""

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a two-dimensional array"
        )

    if len(embeddings) != len(chunks):
        raise ValueError(
            "Embedding count must match chunk count"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embeddings must contain only finite values"
        )

    vector_norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    if not np.allclose(
        vector_norms,
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            "Embeddings must be normalized"
        )

    embeddings_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        embeddings_path,
        embeddings,
        allow_pickle=False,
    )

    metadata = {
        "schema_version": 1,
        "model_name": model_name,
        "chunk_dataset_sha256": (
            compute_chunk_dataset_sha256(chunks)
        ),
        "chunk_count": len(chunks),
        "embedding_dimension": embeddings.shape[1],
        "normalized": True,
        "embeddings_file_sha256": (
            _compute_file_sha256(embeddings_path)
        ),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def load_embedding_artifact(
    chunks: pd.DataFrame,
    expected_model_name: str,
    embeddings_path: Path,
    metadata_path: Path,
) -> np.ndarray:
    """Load embeddings after validating their identity."""

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    if metadata["model_name"] != expected_model_name:
        raise ValueError(
            "Embedding model does not match configuration"
        )

    expected_chunk_hash = (
        compute_chunk_dataset_sha256(chunks)
    )

    if (
        metadata["chunk_dataset_sha256"]
        != expected_chunk_hash
    ):
        raise ValueError(
            "Embedding chunk dataset does not match"
        )

    actual_file_hash = _compute_file_sha256(
        embeddings_path
    )

    if (
        metadata["embeddings_file_sha256"]
        != actual_file_hash
    ):
        raise ValueError(
            "Embedding file SHA-256 does not match metadata"
        )

    embeddings = np.load(
        embeddings_path,
        allow_pickle=False,
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "Loaded embeddings must be two-dimensional"
        )

    if embeddings.shape != (
        metadata["chunk_count"],
        metadata["embedding_dimension"],
    ):
        raise ValueError(
            "Embedding shape does not match metadata"
        )

    if len(embeddings) != len(chunks):
        raise ValueError(
            "Embedding count does not match chunk count"
        )

    return embeddings