"""Experiment: chunked vector index.

Splits each current-version section into overlapping token windows
(default 450 tokens, 64 overlap; bge truncates at 512) with the section
title prepended to every chunk. Section rank at eval time = best chunk rank.

Rationale (measured 2026-10-12): 39.7% of corpus docs exceed bge's 512-token
limit and are silently truncated today (p90 = 1766 tokens, max = 33k).

Usage:
    .venv/bin/python build/build_chunks.py --corpus corpus-src/corpus --out build
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

for _var in ("no_proxy", "NO_PROXY"):
    _val = os.environ.get(_var)
    if _val:
        os.environ[_var] = re.sub(r"\[([0-9a-fA-F:]+)\]", r"\1", _val)

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from build_vectors import parse_front_matter

MODEL = "BAAI/bge-base-en-v1.5"
CHUNK_TOKENS = 450
OVERLAP = 64
BATCH = 64


def chunk_tokens(ids: list[int], size: int, overlap: int) -> list[list[int]]:
    out, i = [], 0
    step = size - overlap
    while i < len(ids):
        out.append(ids[i:i + size])
        if i + size >= len(ids):
            break
        i += step
    return out or [ids]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--chunk-tokens", type=int, default=CHUNK_TOKENS)
    ap.add_argument("--overlap", type=int, default=OVERLAP)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)
    tok = model.tokenizer
    dim = model.get_embedding_dimension()

    corpus = Path(args.corpus)
    out = Path(args.out)
    files = sorted(p for p in corpus.rglob("*.md")
                   if p.name != "index.md" and "--as-of-" not in p.name)
    print(f"section files: {len(files)}", flush=True)

    chunk_texts, chunk_ids, section_of = [], [], []
    for p in files:
        rel = str(p.relative_to(corpus))
        fm, body = parse_front_matter(p.read_text())
        title = fm.get("title", "")
        body_ids = tok.encode(body.strip(), add_special_tokens=False)
        # reserve room for the title in each chunk
        t_ids = tok.encode(title, add_special_tokens=False)
        budget = args.chunk_tokens - len(t_ids) - 8
        for ci, win in enumerate(chunk_tokens(body_ids, budget, args.overlap)):
            text = tok.decode(t_ids + win)
            chunk_texts.append(text)
            cid = f"{rel}#c{ci}"
            chunk_ids.append(cid)
            section_of.append(rel)
    print(f"chunks: {len(chunk_texts)}", flush=True)

    vecs = np.zeros((len(chunk_texts), dim), dtype=np.float32)
    for i in range(0, len(chunk_texts), BATCH):
        e = model.encode(chunk_texts[i:i + BATCH], normalize_embeddings=True,
                         show_progress_bar=False)
        vecs[i:i + len(e)] = e
        if i % (BATCH * 10) == 0:
            print(f"  {min(i + BATCH, len(chunk_texts))}/{len(chunk_texts)}", flush=True)
    np.savez_compressed(out / "chunks.npz", vectors=vecs,
                        chunk_ids=np.array(chunk_ids), sections=np.array(section_of))
    (out / "chunks_manifest.json").write_text(json.dumps({
        "model": args.model, "dims": dim, "chunks": len(chunk_texts),
        "sections": len(files), "chunk_tokens": args.chunk_tokens,
        "overlap": args.overlap, "title_prepended": True,
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, indent=1))
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
