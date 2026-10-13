"""Eval the chunked index on the frozen golden set.

Section rank = best (min) rank across its chunks. Citation fast-path same
as production: exact-citation queries resolve deterministically.

Usage:
    python3 golden/eval_chunks.py --chunks build/chunks.npz --meta build/meta.sqlite
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--golden", default=str(Path(__file__).parent / "eval.json"))
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    z = np.load(args.chunks)
    V, secs = z["vectors"], z["sections"].tolist()
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    con = sqlite3.connect(args.meta)
    model = SentenceTransformer(args.model)
    goldens = json.load(open(args.golden))
    results = []
    for g in goldens:
        q = model.encode([g["query"]], normalize_embeddings=True)[0]
        order = np.argsort(-(Vn @ q))
        seen, ranked_sections = set(), []
        for i in order:
            s = secs[i]
            if s not in seen:
                seen.add(s)
                ranked_sections.append(s)
        m = CITATION_RE.fullmatch(g["query"].strip())
        if m:
            hit = con.execute("SELECT path FROM docs WHERE citation=? AND is_current=1",
                              (f"{m.group(1)} CFR {m.group(2)}",)).fetchone()
            if hit and hit[0] in ranked_sections:
                ranked_sections.remove(hit[0])
                ranked_sections.insert(0, hit[0])
        want = g["expected_path"]
        rank = ranked_sections.index(want) + 1 if want in ranked_sections else 10**9
        results.append({"id": g["id"], "type": g["type"], "rank": rank,
                        "top1": ranked_sections[0]})

    def recall(k):
        return sum(1 for r in results if r["rank"] <= k) / len(results)

    print(f"n={len(results)}  (chunked index, section = best chunk)")
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
