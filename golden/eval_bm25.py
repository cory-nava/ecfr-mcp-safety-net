"""Eval BM25 (FTS5) alone on the frozen golden set.

Usage:
    python3 golden/eval_bm25.py --meta build/meta.sqlite
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path


def fts_query(q: str) -> str:
    terms = re.findall(r"[A-Za-z0-9]+", q.lower())
    return " OR ".join(terms)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--golden", default=str(Path(__file__).parent / "eval.json"))
    args = ap.parse_args()

    con = sqlite3.connect(args.meta)
    goldens = json.load(open(args.golden))
    results = []
    for g in goldens:
        rows = con.execute(
            "SELECT path FROM docs_fts WHERE docs_fts MATCH ? "
            "ORDER BY bm25(docs_fts, 3.0, 1.0)", (fts_query(g["query"]),)).fetchall()
        ranked = [r[0] for r in rows]
        want = g["expected_path"]
        rank = ranked.index(want) + 1 if want in ranked else 10**9
        results.append({"id": g["id"], "type": g["type"], "rank": rank,
                        "top1": ranked[0] if ranked else None})

    def recall(k):
        return sum(1 for r in results if r["rank"] <= k) / len(results)

    print(f"n={len(results)}")
    for k in (1, 3, 5):
        print(f"recall@{k}: {recall(k):.3f}")
    for t in ("exact", "natural", "paraphrase"):
        sub = [r for r in results if r["type"] == t]
        print(f"  {t:10s} n={len(sub):2d}  " +
              "  ".join(f"r@{k}={sum(1 for r in sub if r['rank']<=k)/len(sub):.3f}" for k in (1, 3, 5)))
    print("\nmisses at rank>5:")
    for r in results:
        if r["rank"] > 5:
            g = next(x for x in goldens if x["id"] == r["id"])
            print(f"  #{r['id']} [{r['type']}] rank={r['rank'] if r['rank'] < 10**9 else 'unranked'}")
            print(f"    q: {g['query']}")
            print(f"    want: {g['expected_citation']}  got: {r['top1']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
