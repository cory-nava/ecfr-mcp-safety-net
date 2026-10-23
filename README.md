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
      "env": { "OPENROUTER_API_KEY": "sk-or-v1-..." }
    }
  }
}
```

On Windows the command is `\path\to\ecfr-mcp-safety-net\.venv\Scripts\ecfr-mcp.exe`.
See [Environment variables](#environment-variables) for the full list of tuning options.
`python -m ecfr_mcp` also works.

## Tools

| Tool | What it does |
|---|---|
| `search_corpus(query, top_k=5)` | Ranked sections. An exact citation like `7 CFR 273.9` resolves by direct lookup. |
| `get_section(citation, as_of=None)` | Full section Markdown. `as_of="YYYY-MM-DD"` returns the version in force that day. On a miss, `status` says why: `unknown_citation`, `invalid_date`, `not_in_force` (no current text), `not_in_force_on_date` (a gap in the section's history) or `before_recorded_history`. |
| `version_history(citation)` | Every version with `effective_from` / `effective_to` (last day in force). |
| `list_programs()` | Programs with current section counts. |

Point-in-time answers go back to 2024-01-01: the corpus has every version in
force since then, plus the version already in force on that date.

## Retrieval tiers

The tier is picked from the environment; there are no flags.

| `ECFR_MCP_TIER` | `OPENROUTER_API_KEY` | Tier | Recall@1 / @3 / @5 | Cost |
|---|---|---|---|---|
| unset | unset | **free** | 0.60 / 0.78 / 0.80 | $0, offline |
| unset | set | **best** | ~0.82 / 0.98 / 0.98 | ~$0.002 / query |
| `free` or `best` | either | pinned | | |

- **free:** citation fast-path, then cosine similarity over whole-section
  `bge-small-en-v1.5` vectors. Fully offline after the first run.
- **best:** HyDE candidate generation + LLM listwise rerank over top-50 chunks
  via OpenRouter. If any LLM call fails, that query falls back to free-tier
  ranking and the response includes `"free (degraded from best)"`.

  **How it works:** For each query, one flash call generates a hypothetical
  CFR-style passage (HyDE); its embedding is averaged with the query embedding
  to retrieve better chunk candidates. Five rerank calls then score those
  candidates in parallel using `gpt-6-luna`. Total: 6 API calls per query,
  ~$0.002, latency dominated by Luna's chain-of-thought (~35s/query).

  **To use flash-lite instead of Luna** (faster, ~1–2s/query, 0.76 r@1):
  set `ECFR_MCP_RERANKER=flash-lite`. To disable HyDE: `ECFR_MCP_HYDE=0`.

## Environment variables

All configuration is via environment variables; there are no config files or flags.

| Variable | Default | Effect |
|---|---|---|
| `OPENROUTER_API_KEY` | unset | Enables the best tier (HyDE + LLM rerank). Get a key at openrouter.ai. |
| `ECFR_MCP_TIER` | auto | Pin to `free` or `best`, overriding key detection. |
| `ECFR_MCP_RERANKER` | `luna` | Reranker model. Aliases: `luna` → `openai/gpt-6-luna`, `flash-lite` → `google/gemini-2.5-flash-lite`. Any OpenRouter model ID is accepted. |
| `ECFR_MCP_HYDE` | `1` | Set to `0` to disable HyDE candidate generation and use the plain query vector instead. |
| `ECFR_MCP_HYDE_MODEL` | `google/gemini-2.5-flash-lite` | OpenRouter model for HyDE passage generation. Any cheap, fast chat model works well here. |
| `ECFR_MCP_DATA` | `data/` | Path to the index directory (vectors, corpus, SQLite metadata). |

**Claude Desktop example** with all options shown:

```json
{
  "mcpServers": {
    "ecfr": {
      "command": "/path/to/ecfr-mcp-safety-net/.venv/bin/ecfr-mcp",
      "env": {
        "OPENROUTER_API_KEY": "sk-or-v1-...",
        "ECFR_MCP_RERANKER": "luna",
        "ECFR_MCP_HYDE_MODEL": "google/gemini-2.5-flash-lite"
      }
    }
  }
}
```

## Leaderboard

Numbers on the frozen 50-query golden set (`golden/eval.json`),
`bge-small-en-v1.5` candidates, batches of 10. Full per-query detail:
`golden/results-2026-10-17.md`. Silver set (`silver/eval.json`, 236
queries, 50% paraphrase): best-tier r@1=**0.81** / r@3=0.88 / r@5=0.90
(natural 0.92, paraphrase 0.69); measured 2026-10-23.

| # | Config | r@1 | r@3 | r@5 | LLM calls / query | Cost / 50 queries |
|---|---|---|---|---|---|---|
| 1 | **HyDE + gpt-6-luna** (best-tier default) ✓ | 0.82 | 0.98 | 0.98 | 6 (5 rerank + 1 HyDE) | ~$0.11 |
| 2 | HyDE + gpt-6-luna (pre-cached HyDE passages) † | 0.875 | 1.000 | 1.000 | 6 | $0.107 |
| 3 | Query candidates + gpt-6-luna (no HyDE) | 0.78 | 0.90 | 0.94 | 5 | ~$0.11 |
| 4 | HyDE + flash-lite | 0.76 | 0.92 | 0.94 | 6 | $0.088 |
| 5 | Query candidates + flash-lite | 0.76 | 0.88 | 0.90 | 5 | $0.086 |
| 6 | Dense top-100 + flash-lite | 0.76 | 0.86 | 0.92 | 10 | $0.173 |
| 7 | Hybrid dense∪BM25 + flash-lite | 0.70 | 0.84 | 0.88 | ~9 | $0.143 |
| 8 | Free tier: citation fast-path + vectors | 0.60 | 0.78 | 0.80 | 0 | $0 |
| 9 | HyDE + gpt-5.4-nano | 0.58 | 0.88 | 0.94 | 6 | $0.170 |

✓ Current server default when `OPENROUTER_API_KEY` is set. Paraphrase
r@1 = 0.75 (golden, n=16) / 0.69 (silver, n=118). Rerank batches run in
parallel; total latency is ~35s/query (gated by Luna's chain-of-thought
reasoning). Luna failures fall back to flash-lite, then free tier.

† Row 2 used pre-cached HyDE passages generated by `deepseek-v4-flash`
(`golden/hyde_passages.json`). Row 1 generates passages on-the-fly with
`gemini-2.5-flash-lite`; the 0.055 gap is within noise at n=50 (95% CI
≈ ±0.09) and may also reflect model variance in HyDE quality. The silver
set run will give a stable read.

**Untested candidate:** `bge-reranker-v2-m3`, a self-hosted cross-encoder
(zero API cost; an external 2026-10-18 report puts it at Cohere Rerank
quality on CPU). OOM-killed at 13/50 queries on a 7GB box; needs 16GB+.

Recall is measured with `python golden/eval_server.py [--eval <path>]`,
which runs the server's own retrieval code end to end. `bge-small` was
chosen for its 127 MB download (vs 418 MB for `bge-base-en-v1.5`).
The whitepaper (`whitepaper/ecfr-retrieval-whitepaper.pdf`) and
`DECISION_LOG.md` cover the full experiment history.

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
