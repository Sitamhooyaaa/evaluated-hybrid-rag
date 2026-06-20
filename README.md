# Malaysian Aviation Consumer Protection RAG

> Work in progress!. Document ingestion and development retrieval experiments are complete. Locked evaluation, answer generation, citation evaluation, API development, and deployment are not complete.

## Problem

Malaysian aviation analysts may need to find evidence across consumer-protection codes, amendments, and service-quality reports. Manual search becomes slow when a question requires a specific rule, effective date, report statistic, or evidence from multiple documents.

This project is building an evaluated Retrieval-Augmented Generation (RAG) assistant for evidence-based questions about Malaysian aviation consumer protection and service quality.

The intended user is an aviation consumer-experience or service-quality analyst. This is a public-document portfolio prototype, not an internal Malaysia Aviation Group system and not a source of legal advice.

## Current Scope

Version 1 uses seven official documents:

- Malaysian Aviation Consumer Protection Code 2016
- Malaysian Aviation Consumer Protection Amendment Code 2019
- Malaysian Aviation Consumer Protection Amendment Code 2024
- Consumer Report 2H2023
- Consumer Report 1H2024
- Consumer Report 2H2024
- Consumer Report 1H2025

The corpus currently contains:

- 201 extracted pages
- 186 retrieval-eligible pages
- 368 retrieval chunks

Source PDFs and generated datasets are excluded from Git. Document metadata and source URLs are recorded in the project manifests.

## What Is Implemented

- PDF extraction comparison using PyMuPDF, pdfplumber, and pypdf
- Separate extraction routes for Gazette documents and consumer reports
- English-page selection for bilingual Gazette files
- Repeated-header, disclaimer, and page-number cleaning
- Document metadata and SHA-256 manifests
- Explicit page-exclusion policy
- Reproducible page and chunk JSONL pipelines
- 140-word chunks with 25-word overlap
- Model-token-limit validation
- TF-IDF lexical retrieval baseline
- MiniLM and BGE embedding comparison
- Hybrid retrieval using Reciprocal Rank Fusion
- Page-level Recall@K and Hit@K evaluation
- Gold-evidence validation
- Automated unit and integration tests
- Frozen retrieval configuration before locked evaluation

## Development Results

These results use nine answerable development questions. They were used to make design decisions and are not final test performance.

| Configuration | Recall@1 | Recall@3 | Recall@5 | Hit@5 |
|---|---:|---:|---:|---:|
| TF-IDF, 140-word chunks | 0.389 | 0.778 | 0.778 | 0.778 |
| MiniLM, 140-word chunks | 0.167 | 0.278 | 0.278 | 0.333 |
| BGE, 140-word chunks, query instruction | 0.222 | 0.500 | 0.704 | 0.778 |
| Hybrid TF-IDF and BGE | **0.537** | **0.704** | **0.815** | **0.889** |

The hybrid is the strongest development configuration, but it remains below the provisional Recall@5 target of 0.85.

The configuration is frozen in `evaluation/retrieval_config.json`. If it is changed after inspecting locked results, a new untouched evaluation set will be required.

## Important Findings

### Neural retrieval was not automatically better

TF-IDF substantially outperformed MiniLM. BGE improved semantic retrieval but still did not beat TF-IDF on total gold-page coverage. The result supports keeping a simple lexical baseline instead of assuming embeddings are superior.

### Token limits affected chunking

MiniLM accepts a maximum of 256 tokens. Whole-page chunks exceeded that limit in 51.6% of cases. The selected 140-word strategy produced a maximum of 246 MiniLM tokens and avoided truncation.

### Hybrid retrieval helped but introduced failure modes

Reciprocal Rank Fusion improved average retrieval and recovered evidence missed by individual retrievers. It also suppressed one correct result that TF-IDF ranked second because BGE ranked it poorly.

### Cross-document questions remain difficult

A question combining a report statistic with passenger rights failed because both retrievers focused on the report-related language. Manual query decomposition improved retrieval, but hardcoded subqueries were rejected as a production solution.

Legal provisions continuing across pages also exposed a need for possible adjacent-page expansion or section-aware chunks.

## Project Structure

```text
data/
  manifests/                  Document and page-exclusion metadata
evaluation/
  development_questions.json Development questions and gold pages
  retrieval_config.json      Frozen retrieval configuration
notebooks/
  01_document_extraction_inspection.ipynb
  02_chunking_and_retrieval_evaluation.ipynb
src/aviation_rag/
  ingestion.py               PDF extraction and cleaning
  build_pages.py             Reproducible page dataset pipeline
  chunking.py                Word-window chunking
  build_chunks.py            Reproducible chunk dataset pipeline
  retrieval.py               TF-IDF, semantic, and hybrid retrieval
  evaluation_data.py         Gold validation and retrieval metrics
tests/                        Unit and integration tests
```

## Setup

Python 3.13 is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

The source PDFs are not distributed in this repository. Obtain them from the official URLs recorded in the document manifest and place them in `data/raw/` using the expected filenames.

Build the page and chunk datasets:

```powershell
aviation-rag-build-pages --project-root .
aviation-rag-build-chunks --project-root .
```

Expected current totals:

```text
Pages: 201
Retrieval-eligible pages: 186
Chunks: 368
```

Run tests:

```powershell
python -m pytest -v
```

Current result:

```text
45 passed
```

## Evaluation Design

The planned dataset contains 40 questions:

- 12 development questions for design and tuning
- 28 locked questions for final retrieval evaluation

Question types include:

- direct regulatory lookup
- temporal reasoning
- amendment comparison
- report fact retrieval
- cross-document synthesis
- refusal

Answerable questions use manually verified page-level gold evidence. Refusal questions have no gold pages and will be evaluated later at answer-generation time.

## Current Limitations

- Locked retrieval evaluation has not started.
- Development results come from only nine answerable questions.
- Query decomposition is diagnostic only and not implemented generally.
- Cross-page legal provisions may not retrieve complete evidence.
- No vector database is integrated yet.
- No answer-generation model or prompt is implemented.
- Faithfulness, citation correctness, and refusal quality are not evaluated.
- No FastAPI, user interface, Docker image, or public deployment exists yet.
- This system must not be used as legal advice or as a substitute for official documents.

## Roadmap

1. Create and validate 28 locked evaluation questions without running retrieval.
2. Hash and freeze the locked dataset.
3. Run the frozen retrieval system once and report category-level failures.
4. Implement evidence-grounded answer generation with page citations.
5. Evaluate faithfulness, citation support, and refusal behaviour.
6. Add a vector database only if it solves a demonstrated need.
7. Build a tested API and deployment after evaluation is credible.

## Document Rights

The official source documents are not included in this repository. Users are responsible for reviewing the source websites' access, copyright, and redistribution terms. Code and evaluation artifacts do not replace the official Gazette, regulator, or authority publications.
