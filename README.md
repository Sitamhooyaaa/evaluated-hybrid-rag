# Malaysian Aviation Consumer Protection RAG

> Work in progress. The evaluated RAG pipeline, FastAPI application and local bring-your-own-data Docker workflow are implemented. Final portfolio presentation and audit remain.

## Problem

Malaysian aviation analysts may need evidence from consumer-protection codes, amendments, and service-quality reports. Manual search becomes slow when questions require a specific rule, effective date, historical statistic, or evidence from multiple documents.

This project builds an evaluated Retrieval-Augmented Generation (RAG) assistant for evidence-based questions about Malaysian aviation consumer protection and service quality.

The intended user is an aviation consumer-experience or service-quality analyst. This is a portfolio prototype, not an internal Malaysia Aviation Group system, and not legal advice.

## Publication Model

This repository uses a bring-your-own-data workflow.

Published:

- source code and tests
- document metadata and official source links
- sanitized evaluation questions, rankings and metrics
- notebooks without saved outputs
- retrieval and generation configurations

Not published:

- source PDFs
- extracted pages or chunks
- embeddings
- API keys
- generated-answer files containing retrieved context

Users must obtain documents from the official URLs in `data/manifests/documents.csv`, review the applicable terms, and build derived artifacts locally.

## Local Evaluation Corpus

Version 1 used the Malaysian Aviation Consumer Protection Code 2016, its 2019 and 2024 amendments, and Consumer Reports for 2H2023, 1H2024, 2H2024 and 1H2025.

Local processed totals:

- 201 extracted pages
- 186 retrieval-eligible pages
- 368 chunks

Document IDs, URLs, filenames, access dates, extraction methods and SHA-256 values are recorded in the manifests.

## System Design

```text
Official documents downloaded locally
-> extraction and cleaning
-> page metadata and exclusions
-> 140-word page-bounded chunks
-> TF-IDF retrieval + BGE retrieval
-> weighted reciprocal rank fusion
-> top-five evidence context
-> Gemini generation
-> citation validation
-> FastAPI response
```

Selected retrieval configuration:

- 140 words with 25-word overlap
- `BAAI/bge-small-en-v1.5`
- 20 candidates per retriever
- RRF constant 60
- equal TF-IDF and BGE weights
- final context depth 5

Selected generation configuration:

- Gemini 3.5 Flash
- prompt `dev_v2`
- temperature 0
- thinking budget 0
- maximum output 500 tokens

## Evaluation Design

- 12 development questions for design and prompt comparison
- 28 locked questions for final baseline evaluation
- 23 locked answerable questions
- 5 locked refusal questions

Categories include direct regulatory lookup, temporal reasoning, amendment comparison, report fact retrieval, cross-document synthesis and refusal.

Answerable questions use manually verified page-level gold evidence. The locked dataset was hashed before evaluation.

## Retrieval Results

Development results used nine answerable questions:

| Configuration | Recall@1 | Recall@3 | Recall@5 | Hit@5 |
|---|---:|---:|---:|---:|
| TF-IDF, 140 words | 0.389 | 0.778 | 0.778 | 0.778 |
| MiniLM, 140 words | 0.167 | 0.278 | 0.278 | 0.333 |
| BGE, 140 words | 0.222 | 0.500 | 0.704 | 0.778 |
| Hybrid TF-IDF and BGE | **0.537** | **0.704** | **0.815** | **0.889** |

First locked retrieval run:

- Recall@1: 0.414
- Recall@3: 0.709
- Recall@5: 0.772
- Hit@1: 0.565
- Hit@3: 0.870
- Hit@5: 0.870

Direct regulatory lookup and report retrieval were strong. Cross-document synthesis was the main retrieval failure.

## Answer Results

First locked answer run:

- micro fact coverage: 0.768
- fact-complete rate: 0.696
- full-answer pass rate: 0.609
- citation validity: 0.870
- citation support: 0.957
- faithfulness: 1.000
- correct-refusal rate: 1.000
- clean-refusal pass rate: 0.600

The system was faithful to supplied evidence, but faithfulness did not guarantee completeness. Manual scoring was AI-assisted and reviewed against frozen expected answers, required facts, forbidden inferences, citations and retrieved evidence.

## Main Findings

### Neural retrieval was not automatically better

TF-IDF substantially outperformed MiniLM. BGE improved semantic matching but did not beat TF-IDF on total gold-page coverage. The hybrid performed best because lexical and semantic methods recovered different evidence.

### Token limits changed chunking

Whole-page chunks exceeded MiniLM's token limit in 51.6% of cases. The selected 140-word strategy avoided truncation in the evaluated embedding models.

### Cross-document synthesis remained difficult

Single-query retrieval often found only one intent. Some required pages entered candidate rankings but fell below the top-five context cutoff. Others never entered the candidate set.

### Faithfulness was not enough

Locked answers achieved 1.000 faithfulness but only 0.609 full-answer pass rate. A grounded answer can still be incomplete.

## Project Structure

```text
data/manifests/               Metadata, hashes and exclusions
evaluation/                   Questions, configurations and sanitized metrics
notebooks/                    Exploration and evaluation notebooks 01 to 06
src/aviation_rag/             Reusable ingestion, retrieval, generation and API code
tests/                        Unit and integration tests
```

Important application modules:

- `runtime.py`: loads frozen configuration and builds the answer service
- `api.py`: defines HTTP requests, responses and safety checks
- `main.py`: manages production startup and shutdown

## Local Setup

Python 3.13 is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Download the documents using the URLs and filenames in `data/manifests/documents.csv`, then place them in `data/raw/`.

Build derived datasets:

```powershell
aviation-rag-build-pages --project-root .
aviation-rag-build-chunks --project-root .
aviation-rag-build-embeddings --project-root .
```

Expected totals are 201 pages, 186 eligible pages and 368 chunks.

The embedding command creates a local normalized BGE matrix and metadata under `artifacts/`. The metadata records the model, matrix shape, chunk-dataset SHA-256 and embedding-file SHA-256. Startup rejects stale, reordered, corrupted or wrong-model artifacts.

## Tests

```powershell
python -m pytest -q
```

Current result:

```text
118 passed
```

Tests use temporary files, fake embedding models and fake Gemini clients. They do not require the local corpus or consume API quota.

## Local API

Create an untracked `.env` file:

```text
GEMINI_API_KEY=your_key_here
```

Run:

```powershell
python -m uvicorn aviation_rag.main:app --host 127.0.0.1 --port 8000 --env-file .env
```

Swagger is available at `http://127.0.0.1:8000/docs`.

Endpoints:

- `GET /health`
- `POST /ask`

The API validates input, reuses startup-managed RAG resources, retries temporary provider failures, rejects invalid citation attempts and maps provider failures to HTTP 503.

### Local Runtime Measurement

Local Windows smoke tests measured:

- cold startup: approximately 16 to 20 seconds
- health request: approximately 172 milliseconds through PowerShell
- complete `/ask` request: approximately 2.26 seconds

These are diagnostic measurements from one machine, not performance guarantees. Persisted document embeddings remove repeated encoding of all 368 chunks. The cold start still includes loading the BGE model because new user questions require query embeddings.

### Structured Runtime Logging

The API emits JSON events through Uvicorn's application logger:

- `application_startup_completed` with startup duration
- `http_request_completed` with request ID, method, path, status and duration
- `http_request_failed` with request ID, status, duration and exception type

Logs intentionally exclude raw questions, retrieved evidence, prompts, generated answers and API keys. This preserves operational visibility without creating a second sensitive dataset.

## Local Docker Workflow

The image contains application code and sanitized configuration only. Restricted source PDFs, extracted chunks, embeddings and `.env` are excluded from the build context and image.

Build the CPU-only image:

```powershell
docker build -t malaysia-aviation-rag:local .
```

Start the local bring-your-own-data service:

```powershell
docker compose up -d
```

The Compose service mounts `chunks.jsonl` and the embedding directory read-only, injects `.env` at runtime, and persists the public BGE model cache in a named volume.

Verified container behavior:

- image inspection found no raw documents, processed text, embeddings or `.env`
- health check reached `healthy`
- one grounded answer returned the expected page citation
- one live-data question was refused without citations
- startup failed clearly when artifacts or the API key were missing
- a replacement Compose container reused the model cache

The local image is approximately 2.24 GB because the runtime includes CPU PyTorch and Sentence Transformers. This is acceptable for the v0 local prototype but unsuitable evidence of an optimized production image.

Stop the service without deleting the model-cache volume:

```powershell
docker compose down
```

## Reproducibility and Integrity

- raw and derived corpus files are excluded from Git
- document hashes detect source changes
- prompt SHA-256 validation detects prompt changes
- the locked dataset is hashed
- retrieval configuration was frozen before locked evaluation
- notebook outputs are cleared before publication
- ranking artifacts retain IDs, pages, scores and labels without text
- persisted embeddings are validated against the ordered chunk dataset and embedding-file hashes

## Limitations

- document use and redistribution terms must be reviewed by each user
- BGE model loading still produces an approximately 20-second local cold start
- Gemini quota can make generation unavailable
- no authentication, rate limiting or load testing
- logging is local and has no centralized retention or monitoring backend
- local Docker image is approximately 2.24 GB
- cross-document synthesis remains weak
- cross-page provisions can lose context
- manual scoring contains reviewer judgment
- no public corpus or hosted endpoint is provided
- this system must not replace official documents or legal advice

## Remaining Work

1. Add an architecture diagram and portfolio screenshots.
2. Complete the final repository and container audit.
3. Decide whether a private deployment target is justified by interview needs and document rights.

## Document Rights

The official documents and derived corpus are not distributed in this repository.

Public availability does not automatically grant redistribution permission. Users are responsible for reviewing source-site terms, document notices, copyright and permitted use before processing documents.

This repository publishes an engineering workflow and sanitized evaluation evidence. It does not grant rights to third-party documents.
