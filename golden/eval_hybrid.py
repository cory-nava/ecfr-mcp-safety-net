"""Eval the production retrieval system on the frozen golden set.

System: citation fast-path + vector ranking (no BM25 fusion).
- If the query IS a citation ("7 CFR 273.9"), resolve it deterministically
  against current docs and rank it first.
- Otherwise rank by cosine similarity over bge-base-en-v1.5 vectors.

The BM25 leg exists in meta.sqlite (docs_fts) as a fallback/debug tool.
RRF fusion of BM25+vectors was measured on this same set (k=5..60) and did
not beat vectors+citation (best RRF: 0.48/0.76/0.82 vs 0.64/0.78/0.84),
so it is not part of the production path. See DECISION_LOG.md.

Usage:
    python3 golden/eval_hybrid.py --vectors build/vectors.npz --meta build/meta.sqlite
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

import numpy as np

CITATION_RE = re.compile(r"(\d+)\s+CFR\s+([\w.]+)", re.IGNORECASE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--golden", default=str(Path(__file__).parent / "eval.json"))
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    vec = np.load(args.vectors)
    V, paths = vec["vectors"], vec["paths"].tolist()
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)

    con = sqlite3.connect(args.meta)
    model = SentenceTransformer(args.model)
    goldens = json.load(open(args.golden))
    results = []
    for g in goldens:
        q = model.encode([g["query"]], normalize_embeddings=True)[0]
        ranked = list(np.array(paths)[np.argsort(-(Vn @ q))])
        m = CITATION_RE.fullmatch(g["query"].strip())
        if m:
            cite = f"{m.group(1)} CFR {m.group(2)}"
            hit = con.execute(
                "SELECT path FROM docs WHERE citation = ? AND is_current = 1",
                (cite,)).fetchone()
            if hit and hit[0] in ranked:
                ranked.remove(hit[0])
                ranked.insert(0, hit[0])
        want = g["expected_path"]
        rank = ranked.index(want) + 1 if want in ranked else 10**9
        results.append({"id": g["id"], "type": g["type"], "rank": rank, "top1": ranked[0]})

    def recall(k):
        return sum(1 for r in results if r["rank"] <= k) / len(results)

    print(f"n={len(results)}  (citation fast-path + vectors)")
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
            print(f"  #{r['id']} [{r['type']}] rank={r['rank']}")
            print(f"    q: {g['query']}")
            print(f"    want: {g['expected_citation']}  got: {r['top1']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
