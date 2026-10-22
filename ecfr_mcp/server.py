"""ecfr-mcp: local stdio MCP server over the eCFR Markdown corpus.

Tier selection is environmental (see retrieval.resolve_tier):
  ECFR_MCP_TIER=free|best     pinned tier
  OPENROUTER_API_KEY set      best tier (LLM rerank)
  otherwise                   free tier (citation fast-path + vectors)

Data dir: ECFR_MCP_DATA, or data/ at the repository root.

Tools:
  search_corpus(query, top_k)      ranked sections (tier-appropriate)
  get_section(citation, as_of)     full section text, optionally historical
  version_history(citation)        every version with effective dates
  list_programs()                  programs with current section counts
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from . import retrieval

DATA_DIR = Path(os.environ.get("ECFR_MCP_DATA",
                               Path(__file__).parent.parent / "data"))

mcp = MCPServer("ecfr-mcp")
_corpus: retrieval.Corpus | None = None
_tier: str = "free"
_tier_reason: str = ""


def corpus() -> retrieval.Corpus:
    global _corpus
    if _corpus is None:
        _corpus = retrieval.Corpus(DATA_DIR)
    return _corpus


def _hit_dict(h: retrieval.Hit) -> dict:
    return {"rank": h.rank, "citation": h.citation, "title": h.title,
            "program": h.program, "path": h.path, "score": round(h.score, 4)}


@mcp.tool(description="Search the eCFR corpus. Exact citations like '7 CFR 273.9' "
                       "resolve by direct lookup; everything else is ranked retrieval. "
                       "Uses the LLM rerank tier when OPENROUTER_API_KEY is set.")
async def search_corpus(query: str, top_k: int = 5) -> str:
    top_k = max(1, min(int(top_k), 20))
    if _tier == "best":
        hits, degraded = retrieval.search_best(corpus(), query, top_k)
    else:
        hits, degraded = retrieval.search_free(corpus(), query, top_k), False
    return json.dumps({
        "tier": _tier if not degraded else "free (degraded from best)",
        "tier_reason": _tier_reason,
        "query": query,
        "results": [_hit_dict(h) for h in hits],
    }, indent=1)


@mcp.tool(description="Get the full text of one CFR section. "
                       "as_of='YYYY-MM-DD' (exactly that format) returns the version in force on "
                       "that date. When nothing is found, `status` says why: unknown_citation, "
                       "invalid_date, not_in_force (no current text: repealed / [Reserved] / "
                       "renumbered), not_in_force_on_date (a gap in the section's history), or "
                       "before_recorded_history (earlier than the corpus records; not evidence "
                       "the section was absent).")
async def get_section(citation: str, as_of: str | None = None) -> str:
    return json.dumps(retrieval.get_section(DATA_DIR, citation, as_of), indent=1)


@mcp.tool(description="List every version of a CFR section with effective dates.")
async def version_history(citation: str) -> str:
    return json.dumps(retrieval.version_history(DATA_DIR, citation), indent=1)


@mcp.tool(description="List benefit programs in the corpus with section counts.")
async def list_programs() -> str:
    return json.dumps(retrieval.list_programs(DATA_DIR), indent=1)


async def main() -> None:
    global _tier, _tier_reason
    _tier, _tier_reason = retrieval.resolve_tier()
    # fail fast on missing data, not on first tool call
    if not (DATA_DIR / "meta.sqlite").exists():
        sys.exit(f"[ecfr-mcp] no index data in {DATA_DIR}. Install from a clone of the repo "
                 "(pip install -e .) or point ECFR_MCP_DATA at its data/ directory.")
    corpus()
    print(f"[ecfr-mcp] tier={_tier} ({_tier_reason}) data={DATA_DIR}",
          file=sys.stderr, flush=True)
    await mcp.run_stdio_async()


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
