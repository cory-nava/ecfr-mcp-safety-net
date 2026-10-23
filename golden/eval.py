"""Run the golden eval set against a built vector DB.

Usage:
    python3 golden/eval.py --vectors data/vectors.npz          # vectors only
    python3 golden/eval.py --vectors data/vectors.npz --cite   # + citation fast-path = the free tier

Reports recall@1 / recall@3 / recall@5 overall and per query type.
A query is correct if the expected citation's file ranks in the top-k
by cosine similarity. The query encoder must be the same model that
built the vectors (recorded in manifest.json; used by default).
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", required=True, help="vectors.npz from build_vectors.py")
    ap.add_argument("--golden", default=str(Path(__file__).parent / "eval.json"))
    ap.add_argument("--model", default=None, help="query encoder (default: the model in manifest.json next to the index)")
    ap.add_argument("--query-prompt", default=None,
                    help="prompt_name for query encoding (e.g. 'SearchQuery' for google/embeddinggemma-2)")
    ap.add_argument("--cite", action="store_true",
                    help="resolve exact-citation queries by direct lookup first, as the server's free tier does")
    args = ap.parse_args()
    if args.model is None:
        args.model = json.loads((Path(args.vectors).parent / "manifest.json").read_text(encoding="utf-8"))["model"]

    from sentence_transformers import SentenceTransformer

    vec = np.load(args.vectors)
    V, paths = vec["vectors"], vec["paths"].tolist()
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    path_to_idx = {p: i for i, p in enumerate(paths)}

    model = SentenceTransformer(args.model, trust_remote_code=True)
    cite_re = re.compile(r"(\d+)\s+CFR\s+([\w.]+)", re.IGNORECASE)  # same as ecfr_mcp.retrieval.CITATION_RE
    by_cite = {}
    if args.cite:
        with sqlite3.connect(Path(args.vectors).parent / "meta.sqlite") as con:
            by_cite = dict(con.execute("SELECT citation, path FROM docs WHERE is_current=1"))
        con.close()

    goldens = json.load(open(args.golden))
    results = []
    for g in goldens:
        encode_kw = {"normalize_embeddings": True}
        if args.query_prompt:
            encode_kw["prompt_name"] = args.query_prompt
        q = model.encode([g["query"]], **encode_kw)[0]
        sims = Vn @ q
        ranked = np.argsort(-sims)
        want = g["expected_path"]
        if want not in path_to_idx:
            print(f"SKIP {g['id']}: {want} not in vector DB", file=sys.stderr)
            continue
        m = cite_re.fullmatch(g["query"].strip())
        hit = by_cite.get(f"{m.group(1)} CFR {m.group(2)}") if m else None
        if hit in path_to_idx:  # citation fast-path: the looked-up section goes first
            ranked = np.concatenate([[path_to_idx[hit]], ranked[ranked != path_to_idx[hit]]])
        rank = int(np.where(ranked == path_to_idx[want])[0][0]) + 1
        got = paths[ranked[0]]
        results.append({"id": g["id"], "type": g["type"], "program": g["program"],
                        "rank": rank, "top1": got})

    def recall(k):
        return sum(1 for r in results if r["rank"] <= k) / len(results)

    print(f"model={args.model} cite={args.cite} n={len(results)}")
    for k in (1, 3, 5):
        print(f"recall@{k}: {recall(k):.3f}")
    print("\nby query type:")
    for t in ("exact", "natural", "paraphrase"):
        sub = [r for r in results if r["type"] == t]
        if sub:
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
