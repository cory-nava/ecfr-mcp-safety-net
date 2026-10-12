# ecfr-mcp

An MCP server over the eCFR-as-Markdown benefit-regulations corpus
([ctrimm/ecfr-markdown](https://github.com/ctrimm/ecfr-markdown)): one file
per CFR section, with effective dates and version history.

## Why this exists

The corpus is static files — great for bulk RAG, but awkward for an agent
that needs to *quote the rule*. The MCP server gives agents semantic tools
instead of raw file access: the agent never needs to know the corpus layout,
and temporal questions ("what did the rule say in June 2024?") become
one-liners instead of research projects.

## Use cases

- **Caseworker agent lookup.** "Client has $2,100/mo income, household of
  3 — what does SNAP say?" The agent pulls 7 CFR 273.9, reads the actual
  section, and cites it. This is the primary lane: AI assistants working
  inside case-management tools need citation-grade regulatory answers.
- **Temporal questions.** "What did the SSI rule say when this determination
  was made in March 2025?" The `as_of` tool answers what a current-only
  mirror cannot.
- **"What changed?" analysis.** A policy analyst or advocate asks what
  changed in a part between two dates; the server diffs the version chain.
- **Citation-grade RAG without infra.** A small legal-aid or nonprofit team
  gets grounded regulatory answers through their agent without standing up
  a vector database — BM25-first retrieval, no embedding pipeline required.
- **Eval fixture generation.** Pulling gold-standard section text
  programmatically for policy-grounded LLM evals (e.g. CivBench-style).

## Design

- **Corpus is decoupled from the server.** The repo stays tiny. At setup,
  the server seeds a local cache from a versioned, checksummed release zip
  of the corpus — one download, then fully offline. Refresh is explicit,
  never background magic.
- **Retrieval: citation fast-path + vectors.** Queries that are citations
  ("7 CFR 273.9") resolve deterministically; everything else ranks by
  bge-base-en-v1.5 cosine similarity. Measured on the frozen 50-query
  golden set: recall@1 0.64 / @3 0.78 / @5 0.84. A BM25 index (SQLite FTS5)
  is built and kept as a fallback/debug leg, but RRF fusion was measured
  (k=5..60) and never beat vectors+citation, so it stays out of the
  ranking path. See `golden/` for the eval set and `DECISION_LOG.md` for
  the evidence.
- **Version-aware.** Every section carries `effective_from` /
  `effective_to`; version chains are gapless, so "as of date X" is exact.

## Layout

- `golden/` — 50-query golden eval set (frozen 2026-10-12, before any
  retrieval was built) + eval harness. Measures recall@1/3/5.
- `build/` — vector DB build script (`build_vectors.py`), SQLite metadata,
  build manifest with model/corpus provenance.
- `server/` — the MCP server (planned).

## Status

Golden eval set: done. Vector DB: building. MCP server: planned.
