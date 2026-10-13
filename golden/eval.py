"""Run the golden eval set against a built vector DB.

Usage:
    python3 golden/eval.py --vectors build/vectors.npz --meta build/meta.sqlite

Reports recall@1 / recall@3 / recall@5 overall and per query type.
A query is correct if the expected citation's file ranks in the top-k
by cosine similarity. The query encoder must be the same model that
built the vectors (recorded in build/manifest.json).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

# Sandbox quirk: no_proxy holds bracketed IPv6 literals ([::1]) which make
# httpx (used by huggingface_hub) crash with "Invalid port". Sanitize first.
for _var in ("no_proxy", "NO_PROXY"):
    _val = os.environ.get(_var)
    if _val:
        os.environ[_var] = re.sub(r"\[([0-9a-fA-F:]+)\]", r"\1", _val)

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", required=True, help="vectors.npz from build_vectors.py")
    ap.add_argument("--meta", required=True, help="meta.sqlite from build_vectors.py")
    ap.add_argument("--golden", default=str(Path(__file__).parent / "eval.json"))
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    vec = np.load(args.vectors)
    V, paths = vec["vectors"], vec["paths"].tolist()
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    path_to_idx = {p: i for i, p in enumerate(paths)}

    con = sqlite3.connect(args.meta)
    model = SentenceTransformer(args.model)

    goldens = json.load(open(args.golden))
    results = []
    for g in goldens:
        q = model.encode([g["query"]], normalize_embeddings=True)[0]
        sims = Vn @ q
        ranked = np.argsort(-sims)
        want = g["expected_path"]
        if want not in path_to_idx:
            print(f"SKIP {g['id']}: {want} not in vector DB", file=sys.stderr)
            continue
        rank = int(np.where(ranked == path_to_idx[want])[0][0]) + 1
        got = paths[ranked[0]]
        results.append({"id": g["id"], "type": g["type"], "program": g["program"],
                        "rank": rank, "top1": got})

    def recall(k):
        return sum(1 for r in results if r["rank"] <= k) / len(results)

    print(f"n={len(results)}")
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
