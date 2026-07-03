"""Retrieval methods for aviation RAG documents."""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any


REQUIRED_CHUNK_COLUMNS = {
    "chunk_id",
    "document_id",
    "page_number",
    "chunk_text",
}


class TfidfRetriever:
    """Rank chunks using TF-IDF cosine similarity."""

    def __init__(self, chunks: pd.DataFrame) -> None:
        missing_columns = (
            REQUIRED_CHUNK_COLUMNS - set(chunks.columns)
        )

        if missing_columns:
            raise ValueError(
                "Chunk dataset is missing columns: "
                f"{sorted(missing_columns)}"
            )

        if chunks.empty:
            raise ValueError("Chunk dataset cannot be empty")

        self.chunks = chunks.reset_index(drop=True).copy()

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

        self.chunk_matrix = self.vectorizer.fit_transform(
            self.chunks["chunk_text"]
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> pd.DataFrame:
        """Return the highest-scoring chunks for a query."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_vector = self.vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            self.chunk_matrix,
        ).ravel()

        result_count = min(top_k, len(self.chunks))
        top_indices = (
            scores.argsort()[::-1][:result_count]
        )

        results = self.chunks.iloc[top_indices][
            [
                "chunk_id",
                "document_id",
                "page_number",
                "chunk_text",
            ]
        ].copy()

        results["score"] = scores[top_indices]

        return results.reset_index(drop=True)
    
class SemanticRetriever:
    """Rank chunks using normalized semantic embeddings."""

    def __init__(
        self,
        chunks: pd.DataFrame,
        model: Any,
        query_prefix: str = "",
        batch_size: int = 32,
        chunk_embeddings: np.ndarray | None = None,
    ) -> None:
        missing_columns = (
            REQUIRED_CHUNK_COLUMNS - set(chunks.columns)
        )

        if missing_columns:
            raise ValueError(
                "Chunk dataset is missing columns: "
                f"{sorted(missing_columns)}"
            )

        if chunks.empty:
            raise ValueError("Chunk dataset cannot be empty")

        self.chunks = chunks.reset_index(drop=True).copy()
        self.model = model
        self.query_prefix = query_prefix

        if chunk_embeddings is None:
            self.chunk_embeddings = self.model.encode_document(
                self.chunks["chunk_text"].tolist(),
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        else:
            self.chunk_embeddings = np.asarray(
                chunk_embeddings,
                dtype=float,
            )

            if self.chunk_embeddings.ndim != 2:
                raise ValueError(
                    "Chunk embeddings must be a two-dimensional array"
                )

            if len(self.chunk_embeddings) != len(self.chunks):
                raise ValueError(
                    "Chunk embedding count must match chunk count"
                )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> pd.DataFrame:
        """Return chunks ranked by semantic similarity."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_embedding = self.model.encode_query(
            [self.query_prefix + query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        scores = self.chunk_embeddings @ query_embedding

        result_count = min(top_k, len(self.chunks))
        top_indices = (
            scores.argsort()[::-1][:result_count]
        )

        results = self.chunks.iloc[top_indices][
            [
                "chunk_id",
                "document_id",
                "page_number",
                "chunk_text",
            ]
        ].copy()

        results["score"] = scores[top_indices]

        return results.reset_index(drop=True)
    
class HybridRetriever:
    """Combine lexical and semantic rankings using weighted RRF."""

    def __init__(
        self,
        lexical_retriever: TfidfRetriever,
        semantic_retriever: SemanticRetriever,
        candidate_k: int = 20,
        rrf_constant: int = 60,
        lexical_weight: float = 0.5,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError(
                "candidate_k must be greater than zero"
            )

        if rrf_constant <= 0:
            raise ValueError(
                "rrf_constant must be greater than zero"
            )

        if not 0 <= lexical_weight <= 1:
            raise ValueError(
                "lexical_weight must be between zero and one"
            )

        lexical_ids = set(
            lexical_retriever.chunks["chunk_id"]
        )

        semantic_ids = set(
            semantic_retriever.chunks["chunk_id"]
        )

        if lexical_ids != semantic_ids:
            raise ValueError(
                "Retrievers must use the same chunk dataset"
            )

        self.lexical_retriever = lexical_retriever
        self.semantic_retriever = semantic_retriever
        self.chunks = lexical_retriever.chunks.copy()
        self.candidate_k = candidate_k
        self.rrf_constant = rrf_constant
        self.lexical_weight = lexical_weight
        self.semantic_weight = 1 - lexical_weight

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> pd.DataFrame:
        """Return fused lexical and semantic results."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        lexical_results = (
            self.lexical_retriever.retrieve(
                query=query,
                top_k=self.candidate_k,
            )
        )

        semantic_results = (
            self.semantic_retriever.retrieve(
                query=query,
                top_k=self.candidate_k,
            )
        )

        fused_scores = {}

        for results, weight in [
            (lexical_results, self.lexical_weight),
            (semantic_results, self.semantic_weight),
        ]:
            for rank, chunk_id in enumerate(
                results["chunk_id"],
                start=1,
            ):
                fused_scores[chunk_id] = (
                    fused_scores.get(chunk_id, 0.0)
                    + weight / (
                        self.rrf_constant + rank
                    )
                )

        fused_results = self.chunks.loc[
            self.chunks["chunk_id"].isin(
                fused_scores
            ),
            [
                "chunk_id",
                "document_id",
                "page_number",
                "chunk_text",
            ],
        ].copy()

        fused_results["score"] = (
            fused_results["chunk_id"].map(
                fused_scores
            )
        )

        return (
            fused_results
            .sort_values("score", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )