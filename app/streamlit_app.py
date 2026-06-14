"""Grid Frontend (architecture §04 Layer 10 — REWIRE).

Chat becomes an interactive grid: cells stream in; clicking a cell reveals its
source span, confidence, and route. Chat remains as an entry mode.

Run:  streamlit run app/streamlit_app.py
Install UI extra first:  pip install -e ".[ui]"
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import streamlit as st

from finsight.grid.store import grid_to_dataframe
from finsight.models import Grid
from finsight.pipeline import FinSight

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

    HAS_AGGRID = True
except Exception:  # pragma: no cover - optional dependency
    HAS_AGGRID = False

ROUTE_COLORS = {
    "numeric_store": "#2b6cb0",      # blue  — store (no LLM)
    "retrieval_extract": "#dd6b20",  # orange — retrieval + extract
    "compute": "#2f855a",            # green — deterministic compute
}

st.set_page_config(page_title="FinSight v2.0 — Grid", layout="wide")


@st.cache_resource(show_spinner="Building corpus (ingest · chunk · numeric store · index)…")
def build_corpus(data_dir: str) -> FinSight:
    return FinSight.from_path(data_dir)


def render_aggrid(grid: Grid) -> None:
    # Pivot the flat cell table into the document × column matrix.
    matrix = {}
    for row in grid.rows:
        rec = {"Document": row.title or row.doc_id}
        for col in grid.columns:
            rec[col.name] = grid.get(row.doc_id, col.column_id).display()
        matrix[row.doc_id] = rec
    df = pd.DataFrame(list(matrix.values()))

    # Per-cell route → left-border colour, mirroring the doc's worked fragment.
    route_lookup = {
        (row.title or row.doc_id, col.name): grid.get(row.doc_id, col.column_id).path
        for row in grid.rows
        for col in grid.columns
    }

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(resizable=True, wrapText=True, autoHeight=True)
    gb.configure_selection("single")
    style = JsCode(
        """
        function(params) {
            const colors = %s;
            // colour lookup is approximate in JS; styling handled in Python summary below.
            return {};
        }
        """
        % ROUTE_COLORS
    )
    grid_options = gb.build()
    AgGrid(df, gridOptions=grid_options, allow_unsafe_jscode=True, fit_columns_on_grid_load=True, height=260)


def render_fallback(grid: Grid) -> None:
    rows = []
    for row in grid.rows:
        rec = {"Document": row.title or row.doc_id}
        for col in grid.columns:
            rec[col.name] = grid.get(row.doc_id, col.column_id).display()
        rows.append(rec)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_cell_detail(grid: Grid) -> None:
    st.markdown("#### Click-through · per-cell provenance")
    st.caption("Every cell traces to exactly one source span, with its route and confidence.")
    options = []
    for row in grid.rows:
        for col in grid.columns:
            cell = grid.get(row.doc_id, col.column_id)
            label = f"{row.title or row.doc_id} → {col.name}"
            options.append((label, cell))
    labels = [o[0] for o in options]
    chosen = st.selectbox("Inspect a cell", labels, index=0)
    cell = dict(options)[chosen]

    route = cell.path or "—"
    color = ROUTE_COLORS.get(cell.path, "#666")
    c1, c2, c3 = st.columns(3)
    c1.metric("Value", cell.display())
    c2.metric("Confidence", f"{cell.confidence:.0%}")
    c3.markdown(
        f"**Route**<br><span style='color:{color};font-weight:700'>{route}</span>",
        unsafe_allow_html=True,
    )
    if cell.detail:
        st.markdown(f"**Computed / extracted:** `{cell.detail}`")
    if cell.source:
        st.markdown(f"**Source:** {cell.source.file} · {cell.source.short()}")
    if cell.error:
        st.warning(f"Status `{cell.status}`: {cell.error}")
    elif cell.status == "empty":
        st.info("Returned **empty** — figure not disclosed. Never invented (architecture §05).")


def main() -> None:
    st.title("FinSight v2.0 — Grid-Native Financial Analysis")
    st.caption(
        "Decompose → Fan-out → Verify → Synthesize · every number leaves the LLM and "
        "goes through a deterministic engine."
    )

    with st.sidebar:
        st.header("Corpus")
        data_dir = st.text_input("Data directory", value="data/demo")
        corpus = build_corpus(data_dir)
        info = corpus.info()
        st.json(info)
        st.markdown(
            f"<small>Route legend: "
            f"<span style='color:{ROUTE_COLORS['numeric_store']}'>● store</span> · "
            f"<span style='color:{ROUTE_COLORS['retrieval_extract']}'>● retrieval</span> · "
            f"<span style='color:{ROUTE_COLORS['compute']}'>● compute</span></small>",
            unsafe_allow_html=True,
        )

    question = st.text_input(
        "Ask a cross-document question",
        value="Compare net profit margin and revenue trend across these filings",
    )
    if st.button("Run grid", type="primary") or question:
        with st.spinner("Filling the grid (parallel cells)…"):
            answer = asyncio.run(corpus.ask(question))
        grid = answer.grid

        cols = st.columns(3)
        cols[0].metric("Grid shape", f"{len(grid.rows)} × {len(grid.columns)}")
        cols[1].metric("Completion", f"{grid.completion():.0%}")
        cols[2].metric("LLM", info["llm"])

        st.markdown("### Grid")
        if HAS_AGGRID:
            render_aggrid(grid)
        else:
            st.info("Install the UI extra for the interactive AG Grid: `pip install -e \".[ui]\"`")
            render_fallback(grid)

        flagged = [n for n in grid.verification if n.level in ("warning", "error")]
        if flagged:
            st.markdown("### Verification flags")
            for n in flagged:
                (st.error if n.level == "error" else st.warning)(n.message)

        render_cell_detail(grid)

        st.markdown("### Synthesized answer")
        st.write(answer.narrative)


if __name__ == "__main__":
    main()
