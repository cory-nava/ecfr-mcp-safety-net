"""Build the vector DB for the ecfr-markdown corpus.

- Embeds current-version section files only (--as-of- version files are
  recorded in meta.sqlite for direct lookup but excluded from the index,
  so near-duplicate versions don't pollute search).
- Embeds `title + body`; front matter is excluded (no matching on YAML keys).
- Stores L2-normalized vectors in vectors.npz and metadata in meta.sqlite.
- Writes manifest.json with model/corpus provenance for reproducibility.

Rebuilds meta.sqlite from scratch on every run (an existing file is replaced,
never appended to).

Usage:
    python build/build_vectors.py --corpus ../ecfr-markdown/corpus --out data \
        --snapshot "ctrimm/ecfr-markdown @ <commit> (eCFR current as of <date>)"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MODEL = "BAAI/bge-small-en-v1.5"
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
    ap.add_argument("--snapshot", required=True, help="corpus provenance recorded in manifest.json")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    corpus = Path(args.corpus)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in corpus.rglob("*.md") if p.name != "index.md")
    print(f"corpus files: {len(files)}")

    docs, metas = [], []
    for p in files:
        rel = p.relative_to(corpus).as_posix()
        fm, body = parse_front_matter(p.read_text(encoding="utf-8"))
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
            # No instruction prefixes: bge-*-en-v1.5 was not trained with
            # query:/passage: prefixes; adding them degrades retrieval badly
            # (golden eval 2026-10-12: exact-citation queries ranked ~1270th).
            docs.append((rel, f"{fm.get('title', '')}\n{body.strip()}"))

    print(f"indexing {len(docs)} current-version docs "
          f"({len(files) - len(docs)} version files recorded but not indexed)")

    model = SentenceTransformer(args.model)
    dim = model.get_embedding_dimension()
    paths = [r for r, _ in docs]
    vecs = np.zeros((len(docs), dim), dtype=np.float32)
    for i in range(0, len(docs), BATCH):
        batch = [t for _, t in docs[i:i + BATCH]]
        e = model.encode(batch, normalize_embeddings=True, show_progress_bar=True)
        vecs[i:i + len(batch)] = e
        print(f"  {min(i + BATCH, len(docs))}/{len(docs)}", flush=True)
    np.savez_compressed(out / "vectors.npz", vectors=vecs, paths=np.array(paths))

    db = out / "meta.sqlite"
    db.unlink(missing_ok=True)  # an old table kept by CREATE IF NOT EXISTS is how rows got duplicated
    con = sqlite3.connect(db)
    con.execute("""CREATE TABLE docs
        (path TEXT PRIMARY KEY, citation TEXT, title TEXT, program TEXT, tags TEXT,
         effective_from TEXT, effective_to TEXT, resource TEXT, is_current INTEGER, sha TEXT)""")
    con.executemany(
        "INSERT OR REPLACE INTO docs VALUES (?,?,?,?,?,?,?,?,?,?)",
        [(m["path"], m["citation"], m["title"], m["program"], m["tags"], m["effective_from"],
          m["effective_to"], m["resource"], m["is_current"], m["sha"]) for m in metas])
    con.execute("CREATE INDEX idx_citation ON docs(citation)")
    con.execute("CREATE INDEX idx_current ON docs(is_current)")
    con.commit()
    print("meta rows:", con.execute("SELECT COUNT(*) FROM docs").fetchone()[0])
    con.close()

    manifest = {
        "model": args.model,
        "dims": dim,
        "vectors": len(docs),
        "meta_rows": len(metas),
        "corpus_snapshot": args.snapshot,
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "excluded_from_index": "files with --as-of- in name (earlier versions)",
        "embed_text": "<title>\\n<body>, front matter excluded, no instruction prefixes, bge 512-token truncation applies",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
