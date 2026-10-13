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

- **Corpus ships with the server.** `data/` carries the vector indexes,
  metadata, and corpus Markdown (64MB) — one clone, fully offline, no
  download step. (A versioned release-zip seed remains a possible
  follow-up if the repo weight becomes a problem.)
- **Retrieval tiers are environmental.** No flags: `OPENROUTER_API_KEY`
  set -> best tier (LLM rerank, 0.76/0.94/0.96); unset -> free tier
  (citation fast-path + vectors, 0.64/0.78/0.84, $0). `ECFR_MCP_TIER`
  pins either. LLM failure degrades gracefully to free-tier ranking.
  Measured on the frozen 50-query golden set — see `golden/` and the
  whitepaper (`whitepaper/ecfr-retrieval-whitepaper.pdf`).
- **Version-aware.** Every section carries `effective_from` /
  `effective_to`; "as of date X" is exact via `get_section(as_of=...)`.

## Layout

- `mcp-server/` — the MCP server (`server.py`, `retrieval.py`), 4 tools,
  tier logic, smoke + best-tier tests.
- `data/` — vectors, chunk index + texts, `meta.sqlite`, corpus Markdown.
- `golden/` — 50-query golden eval set (frozen 2026-10-12, before any
  retrieval was built) + eval harness + all seven experiment scripts.
- `build/` — index build scripts, build manifest with model/corpus provenance.
- `whitepaper/` — technical whitepaper (PDF) + charts + generation scripts.

## Status

Retrieval bake-off: done (7 rounds, best 0.76/0.94/0.96). MCP server:
built, smoke-tested (free tier, all 4 tools over stdio), best-tier
tested with stubbed LLM (scoring, degradation, citation bypass).
Whitepaper: PDF in `whitepaper/`, web version in the Artifacts tab.
