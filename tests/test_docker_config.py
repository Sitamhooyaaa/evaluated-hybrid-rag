from pathlib import Path


def test_dockerignore_excludes_sensitive_artifacts() -> None:
    project_root = Path(__file__).resolve().parents[1]

    dockerignore_path = (
        project_root / ".dockerignore"
    )

    ignored_entries = {
        line.strip()
        for line in dockerignore_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
        and not line.strip().startswith("#")
    }

    required_entries = {
        ".git",
        ".venv",
        ".env",
        "data/raw",
        "data/processed",
        "artifacts",
        "notebooks",
    }

    assert required_entries <= ignored_entries

def test_dockerfile_uses_restricted_non_root_runtime() -> None:
    project_root = Path(__file__).resolve().parents[1]

    dockerfile_text = (
        project_root / "Dockerfile"
    ).read_text(encoding="utf-8")

    assert "COPY . ." not in dockerfile_text
    assert "USER app" in dockerfile_text
    assert "HEALTHCHECK" in dockerfile_text

    assert (
        "COPY src ./src"
        in dockerfile_text
    )
    assert (
        "COPY evaluation/retrieval_config.json "
        "./evaluation/"
        in dockerfile_text
    )
    assert (
        "COPY evaluation/generation_config_dev_v2.json "
        "./evaluation/"
        in dockerfile_text
    )

    assert "GEMINI_API_KEY=" not in dockerfile_text

def test_compose_mounts_private_artifacts_at_runtime() -> None:
    project_root = Path(__file__).resolve().parents[1]

    compose_text = (
        project_root / "compose.yaml"
    ).read_text(encoding="utf-8")

    assert "env_file:" in compose_text
    assert ".env" in compose_text

    assert (
        "./data/processed/chunks.jsonl:"
        "/app/data/processed/chunks.jsonl:ro"
        in compose_text
    )
    assert (
        "./artifacts:/app/artifacts:ro"
        in compose_text
    )
    assert (
        "huggingface_cache:"
        "/app/.cache/huggingface"
        in compose_text
    )

    assert "GEMINI_API_KEY:" not in compose_text