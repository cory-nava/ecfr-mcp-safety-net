"""One-time: cache chunk texts to build/chunk_texts.npz so experiment scripts
don't re-tokenize the whole corpus on every run (~4 min saved per run)."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

for _var in ("no_proxy", "NO_PROXY"):
    _val = os.environ.get(_var)
    if _val:
        os.environ[_var] = re.sub(r"\[([0-9a-fA-F:]+)\]", r"\1", _val)

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from build_chunks import chunk_tokens, CHUNK_TOKENS, OVERLAP
from build_vectors import parse_front_matter

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
tok = model.tokenizer
root = Path(__file__).parent.parent
corpus = root / "corpus-src" / "corpus"
files = sorted(p for p in corpus.rglob("*.md")
               if p.name != "index.md" and "--as-of-" not in p.name)
texts, ids = [], []
for i, p in enumerate(files):
    rel = str(p.relative_to(corpus))
    fm, body = parse_front_matter(p.read_text())
    title = fm.get("title", "")
    t_ids = tok.encode(title, add_special_tokens=False)
    budget = CHUNK_TOKENS - len(t_ids) - 8
    body_ids = tok.encode(body.strip(), add_special_tokens=False)
    for ci, win in enumerate(chunk_tokens(body_ids, budget, OVERLAP)):
        ids.append(f"{rel}#c{ci}")
        texts.append(tok.decode(t_ids + win))
    if i % 500 == 0:
        print(f"  {i}/{len(files)} files", flush=True)

z = np.load(root / "build" / "chunks.npz")
assert ids == z["chunk_ids"].tolist(), "misaligned with chunks.npz"
np.savez(root / "build" / "chunk_texts.npz",
         chunk_ids=np.array(ids), texts=np.array(texts, dtype=object))
print(f"saved {len(texts)} chunk texts")
