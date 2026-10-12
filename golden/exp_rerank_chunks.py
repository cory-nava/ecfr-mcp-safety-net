"""Experiment: cross-encoder rerank over CHUNK candidates.

Stage 1: citation fast-path + chunk-vector ranking (top-50 chunks).
Stage 2: bge-reranker-base scores (query, chunk) pairs; section rank = best
chunk rank. Unlike exp_rerank.py (full sections, truncated to 2200 chars),
the reranker here sees the actual passage it scores.

Chunk texts are regenerated deterministically with the same tokenizer and
params as build_chunks.py (fast, no re-embedding) and asserted to align
with chunks.npz before scoring.

Usage:
    python3 golden/exp_rerank_chunks.py --chunks build/chunks.npz --meta build/meta.sqlite --corpus corpus-src/corpus
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

for _var in ("no_proxy", "NO_PROXY"):
    _val = os.environ.get(_var)
    if _val:
        os.environ[_var] = re.sub(r"\[([0-9a-fA-F:]+)\]", r"\1", _val)

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "build"))
from build_chunks import chunk_tokens, CHUNK_TOKENS, OVERLAP
from build_vectors import parse_front_matter

CITATION_RE = re.compile(r"(\d+)\s+CFR\s+([\w.]+)", re.IGNORECASE)
TOP_N = 50


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--golden", default=str(Path(__file__).parent / "eval.json"))
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-base")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer, CrossEncoder

    z = np.load(args.chunks)
    V = z["vectors"]
    saved_ids = z["chunk_ids"].tolist()
    sections = z["sections"].tolist()
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)

    # regenerate chunk texts deterministically; assert alignment
    model = SentenceTransformer(args.model)
    tok = model.tokenizer
    ranker = CrossEncoder(args.reranker)
    corpus = Path(args.corpus)
    files = sorted(p for p in corpus.rglob("*.md")
                   if p.name != "index.md" and "--as-of-" not in p.name)
    texts, ids = [], []
    for p in files:
        rel = str(p.relative_to(corpus))
        fm, body = parse_front_matter(p.read_text())
        title = fm.get("title", "")
        t_ids = tok.encode(title, add_special_tokens=False)
        budget = CHUNK_TOKENS - len(t_ids) - 8
        body_ids = tok.encode(body.strip(), add_special_tokens=False)
        for ci, win in enumerate(chunk_tokens(body_ids, budget, OVERLAP)):
            ids.append(f"{rel}#c{ci}")
            texts.append(tok.decode(t_ids + win))
    assert ids == saved_ids, f"chunk regen misaligned: {len(ids)} vs {len(saved_ids)}"
    print(f"chunk texts regenerated and aligned: {len(texts)}", flush=True)

    con = sqlite3.connect(args.meta)
    goldens = json.load(open(args.golden))
    results = []
    for gi, g in enumerate(goldens):
        q = model.encode([g["query"]], normalize_embeddings=True)[0]
        order = np.argsort(-(Vn @ q))
        m = CITATION_RE.fullmatch(g["query"].strip())
        if m:
            hit = con.execute("SELECT path FROM docs WHERE citation=? AND is_current=1",
                              (f"{m.group(1)} CFR {m.group(2)}",)).fetchone()
            if hit:
                results.append({"id": g["id"], "type": g["type"], "rank": 1,
                                "top1": hit[0]})
                continue
        cand_idx = order[:TOP_N]
        scores = ranker.predict([(g["query"], texts[i]) for i in cand_idx])
        best: dict[str, float] = {}
        for i, s in zip(cand_idx, scores):
            sec = sections[i]
            if sec not in best or s > best[sec]:
                best[sec] = float(s)
        ranked = sorted(best, key=lambda s: -best[s])
        want = g["expected_path"]
        rank = ranked.index(want) + 1 if want in ranked else 10**9
        results.append({"id": g["id"], "type": g["type"], "rank": rank,
                        "top1": ranked[0] if ranked else None})
        print(f"  {gi+1}/{len(goldens)} done", flush=True)

    def recall(k):
        return sum(1 for r in results if r["rank"] <= k) / len(results)

    print(f"n={len(results)}  (rerank top-{TOP_N} chunks with {args.reranker})")
    for k in (1, 3, 5):
        print(f"recall@{k}: {recall(k):.3f}")
    for t in ("exact", "natural", "paraphrase"):
        sub = [r for r in results if r["type"] == t]
        print(f"  {t:10s} n={len(sub):2d}  " +
              "  ".join(f"r@{k}={sum(1 for r in sub if r['rank']<=k)/len(sub):.3f}" for k in (1, 3, 5)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
