"""Production FastAPI application entry point."""

import os
import json
import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from google import genai

from aviation_rag.api import create_app
from aviation_rag.runtime import (
    build_runtime_answer_service,
)

logger = logging.getLogger("uvicorn.error")

def create_production_app(
    project_root: Path | None = None,
    client_factory: Callable[..., Any] = genai.Client,
    service_builder: Callable[..., Any] = (
        build_runtime_answer_service
    ),
) -> FastAPI:
    """Create the API with startup-managed RAG resources."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        started_at = time.perf_counter()
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured"
            )

        root = (
            project_root
            or Path(
                os.getenv(
                    "AVIATION_RAG_PROJECT_ROOT",
                    ".",
                )
            )
        ).resolve()

        gemini_client = client_factory(
            api_key=api_key
        )

        app.state.answer_service = service_builder(
            chunks_path=(
                root
                / "data"
                / "processed"
                / "chunks.jsonl"
            ),
            embeddings_path=(
                root
                / "artifacts"
                / "bge_chunk_embeddings.npy"
            ),
            embedding_metadata_path=(
                root
                / "artifacts"
                / "bge_chunk_embeddings.json"
            ),
            retrieval_config_path=(
                root
                / "evaluation"
                / "retrieval_config.json"
            ),
            generation_config_path=(
                root
                / "evaluation"
                / "generation_config_dev_v2.json"
            ),
            gemini_client=gemini_client,
        )

        startup_duration_ms = round(
            (
                time.perf_counter()
                - started_at
            )
            * 1000,
            3,
        )

        logger.info(
            json.dumps(
                {
                    "event": "application_startup_completed",
                    "duration_ms": startup_duration_ms,
                },
                separators=(",", ":"),
            )
        )

        try:
            yield
        finally:
            close = getattr(
                gemini_client,
                "close",
                None,
            )

            if callable(close):
                close()

    return create_app(lifespan=lifespan)


app = create_production_app()