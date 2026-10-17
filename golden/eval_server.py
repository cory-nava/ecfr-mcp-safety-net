"""Run the golden set through the server's own retrieval code (what actually ships).

Unlike the experiment scripts, this measures ecfr_mcp.retrieval end to end against
data/, including the citation fast-path, with whatever model data/manifest.json names.

Usage:
    python golden/eval_server.py --tier free
    OPENROUTER_API_KEY=... python golden/eval_server.py --tier best
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ecfr_mcp import retrieval  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["free", "best"], default="free")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--golden", default=str(Path(__file__).parent / "eval.json"))
    args = ap.parse_args()

    corpus = retrieval.Corpus(args.data)
    goldens = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    results, degraded = [], 0
    for g in goldens:
        if args.tier == "best":
            hits, deg = retrieval.search_best(corpus, g["query"], top_k=5)
            degraded += deg
        else:
            hits = retrieval.search_free(corpus, g["query"], top_k=5)
        paths = [h.path for h in hits]
        rank = paths.index(g["expected_path"]) + 1 if g["expected_path"] in paths else 99
        results.append({"id": g["id"], "type": g["type"], "rank": rank})

    def recall(rows, k):
        return sum(r["rank"] <= k for r in rows) / len(rows)

    print(f"tier={args.tier} model={corpus.model_name} n={len(results)}"
          + (f" degraded={degraded}" if args.tier == "best" else ""))
    print("overall     " + "  ".join(f"r@{k}={recall(results, k):.2f}" for k in (1, 3, 5)))
    for t in ("exact", "natural", "paraphrase"):
        sub = [r for r in results if r["type"] == t]
        if sub:
            print(f"{t:10s}  " + "  ".join(f"r@{k}={recall(sub, k):.2f}" for k in (1, 3, 5)) + f"  (n={len(sub)})")
    return 1 if degraded else 0


if __name__ == "__main__":
    raise SystemExit(main())
