# FinSight v2.0 — Grid-Native Agentic Analysis for Structured Financial Documents

> *"A grid in the shape of Hebbia, with FinSight's deterministic core intact."*

FinSight v2.0 answers analytical questions that span **many** financial filings by
decomposing each question into a matrix of **documents (rows) × sub-questions
(columns)**, answering every **cell** independently and in parallel — scoped to a
single document — then verifying across cells and synthesizing a cited narrative.

The headline change from v1 is *control flow*, not foundations: the single blended
ReAct loop is replaced by a **planner + many isolated per-cell workers**. The one
capability the market's grid-native tools do not advertise is kept intact — **every
number leaves the LLM and goes through a deterministic Python engine the model
can't touch.**

```
Query → Decompose → Fan-out (parallel cells) → Verify → Synthesize
        rows=docs    each cell scoped to        cross-cell    grid +
        cols=q's     one document               checks        cited answer
```

---

## Why a grid

A blended ReAct loop is excellent for a few documents but leaves performance on the
table when a question spans many filings. The grid buys three things that loop
cannot:

- **Scale** — cells are independent, so the system fans out across thousands of documents.
- **Isolation** — one document per cell, so Company A's text can never contaminate Company B's answer.
- **Audit trail** — every answer traces to exactly one source span.

## The cell — atomic unit of work *and* audit

```python
class Cell:
    doc_id      : str          # the one document this cell reads
    column_id   : str          # which sub-question / field
    question    : str          # the scoped sub-question text
    status      : "pending" | "running" | "done" | "failed" | "empty"
    value       : Any          # answer · figure · computed result
    source      : {file, page, section, char_span}   # exactly one span
    confidence  : float
    path        : "numeric_store" | "retrieval_extract" | "compute"
    cost_tokens : int
```

Each cell takes **one of three routes**:

1. **`numeric_store`** — pull a pre-extracted figure. **No LLM.** The fast path.
2. **`retrieval_extract`** — scoped retrieval (RRF + rerank, hard-filtered to one doc), then extract from the span.
3. **`compute`** — call the deterministic calc engine with exact figures from peer cells.

A missing figure is returned **empty and flagged — never invented** — and any ratio
that depends on it is **refused, not computed on a guess.**

---

## Quickstart (runs offline, no API key)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Inspect the demo corpus and backends
finsight info

# Ask a cross-document question — renders the grid + a cited narrative
finsight ask "Compare net profit margin and revenue trend across these filings"

# Run the evaluation metrics (architecture §09)
finsight eval
```

Without `ANTHROPIC_API_KEY`, FinSight runs in **deterministic mock mode**: the full
ten-layer pipeline still executes end-to-end on the bundled demo dataset
(`data/demo/`) with no network calls. Set the key (copy `.env.example` → `.env`) to
route planning, extraction, and synthesis through Claude with model routing
(Haiku for simple cells, Opus for hard ones / planning / synthesis).

### Interactive grid UI (the doc's recommended fast path)

```bash
pip install -e ".[ui]"
streamlit run app/streamlit_app.py
```

Cells stream in live; click a cell to reveal its source span, confidence, and route.

---

## Architecture — the modified pipeline, layer by layer

| # | Layer | Status | Where in code |
|---|-------|--------|---------------|
| 1 | Format-Aware Ingestion | KEEP | [`ingestion/`](src/finsight/ingestion/ingest.py) |
| 2 | Chunking | KEEP | [`chunking/`](src/finsight/chunking/chunker.py) |
| 3 | Structured Extraction → Numeric Store | KEEP · RE-SCOPED | [`extraction/`](src/finsight/extraction/numeric_store.py) |
| 4 | Dual Indexing + Per-Doc Filter | REWIRE | [`indexing/`](src/finsight/indexing/dual_index.py) |
| 5 | **Query Decomposition & Grid Construction** | **NEW** | [`grid/decompose.py`](src/finsight/grid/decompose.py) |
| 6 | **Per-Cell Execution Engine (Fan-Out)** | **NEW** | [`grid/cell_engine.py`](src/finsight/grid/cell_engine.py) |
| 7 | Scoped Retrieval & Reranking | REWIRE | [`retrieval/`](src/finsight/retrieval/retriever.py) |
| 8 | Deterministic Calculation Engine | KEEP | [`calc/`](src/finsight/calc/engine.py) |
| 9 | **Cross-Cell Verification & Synthesis** | **NEW** | [`grid/verify.py`](src/finsight/grid/verify.py) |
| 10 | Grid Frontend | REWIRE | [`app/streamlit_app.py`](app/streamlit_app.py) |

Cross-cutting: **cell-level caching** ([`cache/`](src/finsight/cache/cell_cache.py)) and the
**evaluation harness** ([`eval/`](src/finsight/eval/metrics.py)) sit across the pipeline.

The pipeline is orchestrated in [`pipeline.py`](src/finsight/pipeline.py):

```python
from finsight.pipeline import FinSight

corpus = FinSight.from_path("data/demo")          # layers 1–4 (+3 numeric store), built once
answer = await corpus.ask("Compare net margin across these filings")  # layers 5–10
print(answer.narrative)
print(answer.grid.completion())                   # grid-completion metric
```

---

## The D × C cost tax (§07) — load-bearing, not polish

A grid issues up to *D documents × C columns* model calls. Four mitigations make
the common case fast and cheap, and all are implemented:

- **Concurrency cap (semaphore)** — bounds in-flight calls (`FINSIGHT_MAX_CONCURRENCY`).
- **Cell-level cache** — a cell is a pure function of `(document, sub-question)`; re-running a grid after one column changes recomputes only that column.
- **Model routing** — simple extraction cells → small model; only hard cells / planning / synthesis → frontier model.
- **Numeric fast path** — cells answerable from the pre-built numeric store skip the LLM entirely.

---

## What stays uniquely FinSight (§08)

The grid shape is borrowed; the **deterministic calculation boundary** is not. Any
cell needing a computed ratio routes through the Python engine
([`calc/engine.py`](src/finsight/calc/engine.py)) with exact figures from the store,
returning **formula, inputs, and result**. Unsupported formulas and missing inputs
are **refused**, never approximated.

> *"We adopted the shape that scales and kept the boundary that makes financial
> answers trustworthy."*

---

## Evaluation (§09)

```bash
finsight eval
```

- **Cell Numeric Accuracy** — fraction of numeric cells equal to ground truth. *The headline number for a financial tool.*
- **Grid Completion** — fraction of cells filled rather than failed/empty.
- **Attribution Correctness** — whether each cell's cited source contains its stated value.
- RAGAS (faithfulness / relevancy / precision / recall) is wired behind the optional `[eval]` extra.

---

## Reference vs. faithful heavy stack

This is a **runnable reference implementation**: the full architecture and all ten
layers are present, but heavy dependencies degrade gracefully so the demo runs with
no GPU or model downloads.

| Concern | Reference fallback (default) | Faithful heavy stack (`pip install -e ".[ml]"`) |
|---|---|---|
| Dense embeddings | hashed bag-of-words (numpy) | **BGE** via sentence-transformers |
| ANN index | brute-force numpy cosine | **FAISS** |
| Reranking | RRF order | **CrossEncoder** |
| Sparse | rank-bm25 | rank-bm25 |
| LLM | deterministic mock | **Claude** (instructor-style structured output) |
| Ingestion | txt / csv | **pdfplumber / PyMuPDF / python-docx** (`[ingest]`) |

Optional extras: `[ml]`, `[ingest]`, `[ui]`, `[eval]`, `[dev]`.

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Project layout

```
src/finsight/
  models.py            # the grid data model (§05): Cell, Grid, GridPlan, Source
  config.py            # settings + optional-dependency capability flags
  pipeline.py          # orchestrates all 10 layers
  cli.py               # finsight info | ask | eval
  llm/                 # tiered structured-output client + offline mock backend
  ingestion/           # layer 1
  chunking/            # layer 2
  extraction/          # layer 3 — numeric store (the cell fast path)
  indexing/            # layer 4 — dual index + per-doc filter
  grid/                # layers 5, 6, 9 — decompose, cell engine, verify/synthesize, store
  retrieval/           # layer 7 — RRF + rerank, scoped
  calc/                # layer 8 — deterministic engine
  cache/               # cell-level cache (§07)
  eval/                # evaluation metrics (§09)
app/streamlit_app.py   # layer 10 — interactive grid (streamlit-aggrid)
data/demo/             # three demo filings + ground truth
tests/                 # unit + integration tests
```

Successor to FinSight v1.0. Internals shown are design shape; exact ranking,
chunking, and prompts remain implementation detail.
