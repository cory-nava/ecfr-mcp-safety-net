# ecfr-mcp server

Local stdio MCP server over the eCFR Markdown corpus
(`ctrimm/ecfr-markdown`, eCFR current as of 2026-10-08).

## Retrieval tiers (environmental — no flags)

| `ECFR_MCP_TIER` | `OPENROUTER_API_KEY` | Tier | Recall@1/@3/@5 | Cost |
|---|---|---|---|---|
| unset | unset | **free** | 0.64 / 0.78 / 0.84 | $0, offline |
| unset | set | **best** | 0.76 / 0.94 / 0.96 | ~$0.002/query |
| `free` / `best` | either | pinned | — | — |

- **free:** deterministic citation lookup (`7 CFR 273.9` -> direct row
  fetch) + `bge-base-en-v1.5` cosine over section vectors.
- **best:** same candidates, then listwise LLM rerank
  (`google/gemini-2.5-flash-lite`, top-50 chunks in batches of 10,
  section = best chunk). Any LLM failure degrades gracefully to
  free-tier ranking for that query instead of erroring.
- Tier numbers are measured on the frozen 50-query golden set;
  see `golden/results-2026-10-12.md` and the whitepaper.

## Tools

- `search_corpus(query, top_k=5)` — ranked sections (tier-appropriate).
- `get_section(citation, as_of=None)` — full section Markdown;
  `as_of="YYYY-MM-DD"` returns the version in force on that date.
- `version_history(citation)` — every version with effective dates.
- `list_programs()` — programs with current section counts.

## Run

```bash
# data/ ships in the repo (vectors, chunks, texts, meta, corpus)
python -m venv .venv && .venv/bin/pip install -r mcp-server/requirements.txt

# free tier
.venv/bin/python mcp-server/server.py

# best tier
OPENROUTER_API_KEY=... .venv/bin/python mcp-server/server.py

# pin a tier
ECFR_MCP_TIER=free .venv/bin/python mcp-server/server.py
```

Data dir override: `ECFR_MCP_DATA=/path/to/data`.

## Tests

```bash
.venv/bin/python mcp-server/test_smoke.py      # MCP stdio, free tier, all 4 tools
.venv/bin/python mcp-server/test_best_tier.py   # rerank scoring, degradation, citation bypass
```

## Files

- `server.py` — MCP stdio server, tool definitions, tier wiring.
- `retrieval.py` — tier resolution, free/best search, section/version/program helpers.
- `requirements.txt` — `mcp>=2`, `numpy`, `sentence-transformers`.
- `../data/` — vectors, chunks, chunk texts, `meta.sqlite`, corpus Markdown.
