"""The v2 pipeline in execution order (architecture §04).

Front half (ingest → chunk → numeric store → dual index) is built once per corpus.
Each query then runs the grid: decompose → fan-out → verify → synthesize.

    corpus = FinSight.from_path("data/demo")     # layers 1-4 (+3 numeric store)
    answer = await corpus.ask("Compare net margin across these filings")  # layers 5-10
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from finsight.cache import CellCache
from finsight.chunking import chunk_document
from finsight.extraction import NumericStore, build_numeric_store
from finsight.grid import CellEngine, decompose, synthesize, verify_grid
from finsight.grid.cell_engine import CellContext
from finsight.indexing import DualIndex
from finsight.ingestion import ParsedDocument, ingest_path
from finsight.llm import LLM, get_llm
from finsight.models import Cell, Grid
from finsight.retrieval import Retriever


@dataclass
class GridAnswer:
    grid: Grid
    narrative: str

    @property
    def completion(self) -> float:
        return self.grid.completion()


class FinSight:
    """A built corpus: the reusable front half of the pipeline."""

    def __init__(
        self,
        docs: list[ParsedDocument],
        *,
        llm: Optional[LLM] = None,
        cache_namespace: str = "cells",
    ):
        self.docs = docs
        self.docs_by_id = {d.doc_id: d for d in docs}
        self.llm = llm or get_llm()

        # Layer 2: chunking.
        self.chunks = [c for d in docs for c in chunk_document(d)]
        # Layer 3: structured extraction → numeric store (the cell fast path).
        self.numeric_store: NumericStore = build_numeric_store(docs)
        # Layer 4: dual indexing + per-doc filter.
        self.index = DualIndex()
        self.index.build(self.chunks)
        # Layer 7: scoped retrieval & reranking.
        self.retriever = Retriever(self.index)
        # Cross-cutting: cell cache.
        self.cache = CellCache(cache_namespace)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_path(cls, path: str | Path, **kw) -> "FinSight":
        # Layer 1: format-aware ingestion.
        docs = ingest_path(path)
        return cls(docs, **kw)

    # -- query -------------------------------------------------------------
    async def ask(
        self,
        question: str,
        *,
        doc_ids: Optional[list[str]] = None,
        on_cell: Optional[Callable[[Cell], None]] = None,
        synthesize_answer: bool = True,
    ) -> GridAnswer:
        scope = (
            [self.docs_by_id[d] for d in doc_ids if d in self.docs_by_id]
            if doc_ids
            else self.docs
        )

        # Layer 5: query decomposition & grid construction.
        plan = await decompose(question, scope, self.llm)
        grid = Grid.from_plan(plan)

        # Layer 6: per-cell execution engine (fan-out).
        ctx = CellContext(
            numeric_store=self.numeric_store,
            retriever=self.retriever,
            llm=self.llm,
            cache=self.cache,
            docs=self.docs_by_id,
        )
        engine = CellEngine(ctx)
        if on_cell:
            engine.on_cell = on_cell
        await engine.execute_grid(grid)

        # Layer 9: cross-cell verification & synthesis.
        verify_grid(grid, self.numeric_store)
        narrative = await synthesize(grid, self.llm) if synthesize_answer else ""
        return GridAnswer(grid=grid, narrative=narrative)

    # -- introspection -----------------------------------------------------
    def info(self) -> dict:
        return {
            "documents": len(self.docs),
            "chunks": len(self.chunks),
            "numeric_facts": len(self.numeric_store),
            "embedding_backend": self.index.backend,
            "reranker_backend": self.retriever.reranker_backend,
            "llm": "live" if self.llm.live else "mock",
        }
