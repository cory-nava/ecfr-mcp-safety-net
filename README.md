# ecfr-mcp-safety-net

An [MCP](https://modelcontextprotocol.io) server over the federal
benefit-program regulations in
[ctrimm/ecfr-markdown-safety-net](https://github.com/ctrimm/ecfr-markdown-safety-net): SNAP, WIC,
school meals, TANF, Medicaid/CHIP, SSI, housing vouchers, LIHEAP, Lifeline and
more, one CFR section per document, with effective dates and version history.

Agents get four tools instead of raw files: search, exact section text,
"what did this rule say on date X?", and version history.

## Quick start

```bash
git clone https://github.com/ctrimm/ecfr-mcp-safety-net && cd ecfr-mcp-safety-net
python -m venv .venv && .venv/bin/pip install -e .
# CPU-only machines can skip the CUDA build of PyTorch: install it first with
#   .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The indexes and the corpus ship in `data/` (29 MB). The first search
downloads the embedding model (`BAAI/bge-small-en-v1.5`, 127 MB) to the
Hugging Face cache; after that the free tier runs fully offline.

**Claude Code**

```bash
claude mcp add ecfr -- /path/to/ecfr-mcp-safety-net/.venv/bin/ecfr-mcp
```

**Claude Desktop and other MCP clients** (`claude_desktop_config.json` or equivalent):

```json
{
  "mcpServers": {
    "ecfr": {
      "command": "/path/to/ecfr-mcp-safety-net/.venv/bin/ecfr-mcp",
      "env": { "OPENROUTER_API_KEY": "optional: enables the best tier" }
    }
  }
}
```

On Windows the command is `\path\to\ecfr-mcp-safety-net\.venv\Scripts\ecfr-mcp.exe`.
`python -m ecfr_mcp` also works.

## Tools

| Tool | What it does |
|---|---|
| `search_corpus(query, top_k=5)` | Ranked sections. An exact citation like `7 CFR 273.9` resolves by direct lookup. |
| `get_section(citation, as_of=None)` | Full section Markdown. `as_of="YYYY-MM-DD"` returns the version in force that day. |
| `version_history(citation)` | Every version with `effective_from` / `effective_to` (last day in force). |
| `list_programs()` | Programs with current section counts. |

Point-in-time answers go back to 2024-01-01: the corpus has every version in
force since then, plus the version already in force on that date.

## Retrieval tiers

The tier is picked from the environment; there are no flags.

| `ECFR_MCP_TIER` | `OPENROUTER_API_KEY` | Tier | Recall@1 / @3 / @5 | Cost |
|---|---|---|---|---|
| unset | unset | **free** | 0.60 / 0.78 / 0.80 | $0, offline |
| unset | set | **best** | 0.76 / 0.88 / 0.90 | ~$0.002 / query |
| `free` or `best` | either | pinned | | |

- **free:** citation fast-path, then cosine similarity over whole-section
  `bge-small-en-v1.5` vectors.
- **best:** the top 50 chunks by vector similarity, reranked listwise by an LLM
  (`google/gemini-2.5-flash-lite` through OpenRouter, batches of 10; a
  section scores as its best chunk). If the LLM call fails, that query falls
  back to free-tier ranking and the response says `"free (degraded from best)"`.

  **Reranker cost (estimated):** 5 scoring calls per query (50 chunks ÷ 10
  per batch) — about **$0.0017 per query** at current flash-lite pricing.
  Measured: $0.0864 for the full 50-query golden run (2026-10-17), zero
  failed calls. Latency is 30–60s sequential, batchable. Whether the
  +16pp r@1 over free is worth it per query is a product call — the
  `ECFR_MCP_TIER=free` pin exists for exactly this tradeoff.

- **HyDE-averaged candidates + rerank** is the best *measured* config
  (0.76 / 0.92 / 0.94) but is **not** the server default: it adds one
  HyDE passage-generation call per query (~$0.0001 on flash, 5–10s) for
  +4pp at r@3/r@5 with r@1 unchanged. The default stays zero extra calls
  beyond the rerank itself. See the leaderboard below.

## Leaderboard

All numbers on the frozen 50-query golden set, `bge-small-en-v1.5`
candidates, measured 2026-10-17. Reranker is `google/gemini-2.5-flash-lite`
unless noted (batches of 10). Detail per query type and per miss:
`golden/results-2026-10-17.md`.

| # | Config | r@1 | r@3 | r@5 | LLM calls / query | Cost / 50 queries |
|---|---|---|---|---|---|---|
| 1 | HyDE-averaged candidates + **gpt-6-luna** rerank ★ | 0.875 | 1.000 | 1.000 | 6 (5 rerank + 1 HyDE) | $0.107 |
| 2 | HyDE-averaged candidates + rerank | 0.76 | 0.92 | 0.94 | 6 (5 rerank + 1 HyDE) | $0.088 |
| 3 | Query candidates + rerank (best-tier default) | 0.76 | 0.88 | 0.90 | 5 | $0.086 |
| 4 | Dense top-100 + rerank | 0.76 | 0.86 | 0.92 | 10 | $0.173 |
| 5 | Hybrid dense∪BM25 candidates + rerank | 0.70 | 0.84 | 0.88 | ~9 | $0.143 |
| 6 | Free tier: citation fast-path + vectors | 0.60 | 0.78 | 0.80 | 0 | $0 |
| 7 | HyDE-averaged candidates + **gpt-5.4-nano** rerank | 0.58 | 0.88 | 0.94 | 6 (5 rerank + 1 HyDE) | $0.170 |

★ Best measured, not the default: Luna scores 48/50 (2 queries returned
unparseable responses) and takes 30–40s/batch — it emits ~80 reasoning
tokens per call, which is the quality edge and the latency cost. The HyDE
call costs ~$0.0001/query (one flash generation); the rerank is ~$0.0017/query,
HyDE+rerank path is ~$0.0018/query. r@1 is identical to the default —
the gain is entirely at r@3/r@5, where the stronger candidate pool
matters. Adopting it as the default is a latency/cost tradeoff, hence
opt-in.

**Untested candidate:** `bge-reranker-v2-m3`, a self-hosted cross-encoder
(zero API cost; an external 2026-10-18 report puts it at Cohere Rerank
quality on CPU). It has no row above because it has never finished a run
on this corpus — its one attempt OOM-killed at 13/50 queries on a 7GB
box. Retrying it needs a 16GB+ box; until then it is a candidate, not a
measured result.

For reference, the 2026-10-12/13 bake-off on `bge-base-en-v1.5`
(768-dim): free 0.64/0.78/0.84, rerank 0.76/0.94/0.96 — see the
whitepaper (`whitepaper/ecfr-retrieval-whitepaper.pdf`).

Recall is measured on the frozen 50-query golden set (`golden/`), with
`python golden/eval_server.py`, which runs the server's own code.
`bge-small` was chosen for its 127 MB download (vs 418 MB for
`bge-base-en-v1.5`, which scores 0.64 / 0.78 / 0.84 on the free tier with
the same corpus and code).
The whitepaper (`whitepaper/ecfr-retrieval-whitepaper.pdf`) and
`DECISION_LOG.md` cover the seven rounds of experiments behind these
choices; they predate the model switch, so their numbers are for
`bge-base`.

## Use cases

- **Caseworker assistants.** "Household of 3, $2,100 a month: what does SNAP
  say?" The agent pulls 7 CFR 273.9, reads it, and cites it.
- **Point-in-time questions.** "What did the SSI rule say when this
  determination was made in March 2025?"
- **What changed?** Compare a section's versions across dates.
- **Grounded answers without infrastructure.** No vector database or hosted
  service to run.
- **Eval fixtures.** Pull gold-standard section text for policy-grounded LLM evals.

## Caveats

This is a convenience copy of the eCFR, not the official legal edition; see
the caveats in [ecfr-markdown-safety-net](https://github.com/ctrimm/ecfr-markdown-safety-net#caveats).
The bundled snapshot is recorded in `data/manifest.json` (`corpus_snapshot`).

## Development

```bash
python -m unittest discover -s tests -t .   # the smoke test also needs the embedding model
```

Rebuild `data/` after refreshing the corpus (needs `sentence-transformers`;
`build/requirements.txt` pins a CPU-only PyTorch):

```bash
python build/build_vectors.py --corpus ../ecfr-markdown-safety-net/corpus --out data \
    --snapshot "ctrimm/ecfr-markdown-safety-net @ <commit> (eCFR current as of <date>)"
python build/build_chunks.py  --corpus ../ecfr-markdown-safety-net/corpus --out data
python build/pack_corpus.py   --corpus ../ecfr-markdown-safety-net/corpus --out data
python golden/eval_server.py --tier free
```

Layout: `ecfr_mcp/` is the server (`server.py`, `retrieval.py`), `data/` the
shipped indexes and corpus, `build/` the index builders, `golden/` the eval set
and experiment scripts, `whitepaper/` the write-up, and `HOSTING.md` covers
RAM, disk, and cloud costs.

## License

Code: MIT (see `LICENSE`). The regulatory text is a work of the U.S. government
and is in the public domain.
