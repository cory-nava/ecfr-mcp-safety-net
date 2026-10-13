"""Build the vector DB for the ecfr-markdown corpus.

- Embeds current-version section files only (--as-of- version files are
  recorded in meta.sqlite for direct lookup but excluded from the index,
  so near-duplicate versions don't pollute search).
- Embeds `title + body`; front matter is excluded (no matching on YAML keys).
- Stores L2-normalized vectors in vectors.npz and metadata in meta.sqlite.
- Writes manifest.json with model/corpus provenance for reproducibility.

Usage:
    .venv/bin/python build/build_vectors.py --corpus corpus-src/corpus --out build
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# Sandbox quirk: no_proxy holds bracketed IPv6 literals ([::1]) which make
# httpx (used by huggingface_hub) crash with "Invalid port". Sanitize first.
for _var in ("no_proxy", "NO_PROXY"):
    _val = os.environ.get(_var)
    if _val:
        os.environ[_var] = re.sub(r"\[([0-9a-fA-F:]+)\]", r"\1", _val)

import numpy as np

MODEL = "BAAI/bge-base-en-v1.5"
BATCH = 64


def parse_front_matter(text: str) -> tuple[dict, str]:
    """Split (front_matter dict, body). Minimal YAML subset: scalars + [lists]."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    fm: dict = {}
    for line in text[4:end].splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        if v.startswith('"') and v.endswith('"'):
            v = v[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        elif v.startswith("[") and v.endswith("]"):
            v = [x.strip().strip('"') for x in v[1:-1].split(",") if x.strip()]
        fm[k.strip()] = v
    return fm, text[end + 5:].lstrip("\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    corpus = Path(args.corpus)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in corpus.rglob("*.md") if p.name != "index.md")
    print(f"corpus files: {len(files)}")

    docs, metas = [], []
    for p in files:
        rel = str(p.relative_to(corpus))
        fm, body = parse_front_matter(p.read_text())
        is_version = "--as-of-" in p.name
        metas.append({
            "path": rel,
            "citation": fm.get("citation", ""),
            "title": fm.get("title", ""),
            "program": (fm.get("tags") or [""])[0] if isinstance(fm.get("tags"), list) else "",
            "tags": ",".join(fm.get("tags", []) if isinstance(fm.get("tags"), list) else []),
            "effective_from": fm.get("effective_from", ""),
            "effective_to": fm.get("effective_to", ""),
            "resource": fm.get("resource", ""),
            "is_current": 0 if is_version else 1,
            "sha": hashlib.sha256(p.read_bytes()).hexdigest()[:16],
        })
        if not is_version:
            # No instruction prefixes: bge-base-en-v1.5 was not trained with
            # query:/passage: prefixes; adding them degrades retrieval badly
            # (golden eval 2026-10-12: exact-citation queries ranked ~1270th).
            docs.append((rel, f"{fm.get('title', '')}\n{body.strip()}"))

    print(f"indexing {len(docs)} current-version docs "
          f"({len(files) - len(docs)} version files recorded but not indexed)")

    model = SentenceTransformer(args.model)
    dim = model.get_sentence_embedding_dimension()
    paths = [r for r, _ in docs]
    vecs = np.zeros((len(docs), dim), dtype=np.float32)
    for i in range(0, len(docs), BATCH):
        batch = [t for _, t in docs[i:i + BATCH]]
        e = model.encode(batch, normalize_embeddings=True, show_progress_bar=True)
        vecs[i:i + len(batch)] = e
        print(f"  {min(i + BATCH, len(docs))}/{len(docs)}", flush=True)
    np.savez_compressed(out / "vectors.npz", vectors=vecs, paths=np.array(paths))

    con = sqlite3.connect(out / "meta.sqlite")
    con.execute("""CREATE TABLE IF NOT EXISTS docs
        (path TEXT PRIMARY KEY, citation TEXT, title TEXT, program TEXT, tags TEXT,
         effective_from TEXT, effective_to TEXT, resource TEXT, is_current INTEGER, sha TEXT)""")
    con.executemany(
        "INSERT OR REPLACE INTO docs VALUES (?,?,?,?,?,?,?,?,?,?)",
        [(m["path"], m["citation"], m["title"], m["program"], m["tags"], m["effective_from"],
          m["effective_to"], m["resource"], m["is_current"], m["sha"]) for m in metas])
    con.execute("CREATE INDEX IF NOT EXISTS idx_citation ON docs(citation)")
    con.commit()
    print("meta rows:", con.execute("SELECT COUNT(*) FROM docs").fetchone()[0])
    con.close()

    manifest = {
        "model": args.model,
        "dims": dim,
        "vectors": len(docs),
        "meta_rows": len(metas),
        "corpus_snapshot": "ctrimm/ecfr-markdown @ main, retrieved 2026-10-12 (eCFR current as of 2026-10-08)",
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "excluded_from_index": "files with --as-of- in name (earlier versions)",
        "embed_text": "<title>\\n<body>, front matter excluded, no instruction prefixes, bge 512-token truncation applies",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
