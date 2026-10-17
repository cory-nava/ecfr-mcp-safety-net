# ecfr-mcp

An [MCP](https://modelcontextprotocol.io) server over the federal
benefit-program regulations in
[ctrimm/ecfr-markdown](https://github.com/ctrimm/ecfr-markdown): SNAP, WIC,
school meals, TANF, Medicaid/CHIP, SSI, housing vouchers, LIHEAP, Lifeline and
more, one CFR section per document, with effective dates and version history.

Agents get four tools instead of raw files: search, exact section text,
"what did this rule say on date X?", and version history.

## Quick start

```bash
git clone https://github.com/ctrimm/ecfr-mcp && cd ecfr-mcp
python -m venv .venv && .venv/bin/pip install -e .
# CPU-only machines can skip the CUDA build of PyTorch: install it first with
#   .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The indexes and the corpus ship in `data/` (29 MB). The first search
downloads the embedding model (`BAAI/bge-small-en-v1.5`, 127 MB) to the
Hugging Face cache; after that the free tier runs fully offline.

**Claude Code**

```bash
claude mcp add ecfr -- /path/to/ecfr-mcp/.venv/bin/ecfr-mcp
```

**Claude Desktop and other MCP clients** (`claude_desktop_config.json` or equivalent):

```json
{
  "mcpServers": {
    "ecfr": {
      "command": "/path/to/ecfr-mcp/.venv/bin/ecfr-mcp",
      "env": { "OPENROUTER_API_KEY": "optional: enables the best tier" }
    }
  }
}
```

On Windows the command is `\path\to\ecfr-mcp\.venv\Scripts\ecfr-mcp.exe`.
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
| unset | set | **best** | 0.76 / 0.94 / 0.96 † | ~$0.002 / query |
| `free` or `best` | either | pinned | | |

- **free:** citation fast-path, then cosine similarity over whole-section
  `bge-small-en-v1.5` vectors.
- **best:** the top 50 chunks by vector similarity, reranked listwise by an LLM
  (`google/gemini-2.5-flash-lite` through OpenRouter, batches of 10; a
  section scores as its best chunk). If the LLM call fails, that query falls
  back to free-tier ranking and the response says `"free (degraded from best)"`.

Recall is measured on the frozen 50-query golden set (`golden/`), with
`python golden/eval_server.py`, which runs the server's own code.
`bge-base-en-v1.5` scores 0.64 / 0.78 / 0.84 on the free tier with the same
corpus and code; `bge-small` was chosen for its 127 MB download (vs 418 MB).
† The best-tier figure was measured during the bake-off with `bge-base`
candidates and hasn't been re-measured since the switch to `bge-small` (it needs
an OpenRouter key: `OPENROUTER_API_KEY=... python golden/eval_server.py --tier best`).
The whitepaper (`whitepaper/ecfr-retrieval-whitepaper.pdf`) and
`DECISION_LOG.md` cover the seven rounds of experiments behind these choices;
they predate the model switch, so their numbers are for `bge-base`.

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
the caveats in [ecfr-markdown](https://github.com/ctrimm/ecfr-markdown#caveats).
The bundled snapshot is recorded in `data/manifest.json` (`corpus_snapshot`).

## Development

```bash
python -m unittest discover -s tests -t .   # the smoke test also needs the embedding model
```

Rebuild `data/` after refreshing the corpus (needs `sentence-transformers`;
`build/requirements.txt` pins a CPU-only PyTorch):

```bash
python build/build_vectors.py --corpus ../ecfr-markdown/corpus --out data \
    --snapshot "ctrimm/ecfr-markdown @ <commit> (eCFR current as of <date>)"
python build/build_chunks.py  --corpus ../ecfr-markdown/corpus --out data
python build/pack_corpus.py   --corpus ../ecfr-markdown/corpus --out data
python golden/eval_server.py --tier free
```

Layout: `ecfr_mcp/` is the server (`server.py`, `retrieval.py`), `data/` the
shipped indexes and corpus, `build/` the index builders, `golden/` the eval set
and experiment scripts, `whitepaper/` the write-up, and `HOSTING.md` covers
RAM, disk, and cloud costs.

## License

Code: MIT (see `LICENSE`). The regulatory text is a work of the U.S. government
and is in the public domain.
