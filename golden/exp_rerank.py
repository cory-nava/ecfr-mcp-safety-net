"""Experiment: cross-encoder rerank on top of the production system.

Stage 1: citation fast-path + vector ranking (top-50 candidates).
Stage 2: BAAI/bge-reranker-base scores (query, title+body) pairs, re-sorts.
Exact-citation queries keep their deterministic rank-1 (not reranked).

Usage:
    python3 golden/exp_rerank.py --vectors build/vectors.npz --meta build/meta.sqlite --corpus corpus-src/corpus
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from pathlib import Path

for _var in ("no_proxy", "NO_PROXY"):
    _val = os.environ.get(_var)
    if _val:
        os.environ[_var] = re.sub(r"\[([0-9a-fA-F:]+)\]", r"\1", _val)

import numpy as np

CITATION_RE = re.compile(r"(\d+)\s+CFR\s+([\w.]+)", re.IGNORECASE)
TOP_N = 50

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "build"))
from build_vectors import parse_front_matter


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--golden", default=str(Path(__file__).parent / "eval.json"))
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-base")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer, CrossEncoder

    vec = np.load(args.vectors)
    V, paths = vec["vectors"], vec["paths"].tolist()
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    con = sqlite3.connect(args.meta)
    model = SentenceTransformer(args.model)
    ranker = CrossEncoder(args.reranker)
    corpus = Path(args.corpus)
    texts = {}
    for p in paths:
        fm, body = parse_front_matter((corpus / p).read_text())
        texts[p] = f"{fm.get('title', '')}\n{body.strip()}"[:2200]

    goldens = json.load(open(args.golden))
    results = []
    for gi, g in enumerate(goldens):
        q = model.encode([g["query"]], normalize_embeddings=True)[0]
        ranked = list(np.array(paths)[np.argsort(-(Vn @ q))])
        m = CITATION_RE.fullmatch(g["query"].strip())
        direct = None
        if m:
            hit = con.execute("SELECT path FROM docs WHERE citation=? AND is_current=1",
                              (f"{m.group(1)} CFR {m.group(2)}",)).fetchone()
            direct = hit[0] if hit else None
        if direct:
            results.append({"id": g["id"], "type": g["type"], "rank": 1, "top1": direct})
            continue
        cands = ranked[:TOP_N]
        scores = ranker.predict([(g["query"], texts[p]) for p in cands])
        reranked = [p for _, p in sorted(zip(scores, cands), reverse=True)]
        want = g["expected_path"]
        rank = reranked.index(want) + 1 if want in reranked else 10**9
        results.append({"id": g["id"], "type": g["type"], "rank": rank, "top1": reranked[0]})
        print(f"  {gi+1}/{len(goldens)} done", flush=True)

    def recall(k):
        return sum(1 for r in results if r["rank"] <= k) / len(results)

    print(f"n={len(results)}  (rerank top-{TOP_N} with {args.reranker})")
    for k in (1, 3, 5):
        print(f"recall@{k}: {recall(k):.3f}")
    for t in ("exact", "natural", "paraphrase"):
        sub = [r for r in results if r["type"] == t]
        print(f"  {t:10s} n={len(sub):2d}  " +
              "  ".join(f"r@{k}={sum(1 for r in sub if r['rank']<=k)/len(sub):.3f}" for k in (1, 3, 5)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
