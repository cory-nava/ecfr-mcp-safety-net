"""Add a BM25 (SQLite FTS5) index to meta.sqlite.

Indexes title + body of current-version docs only (front matter excluded,
same as the vector build). Title gets 3x weight at query time via
bm25(docs_fts, 3.0, 1.0).

Usage:
    .venv/bin/python build/build_bm25.py --corpus corpus-src/corpus --meta build/meta.sqlite
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_vectors import parse_front_matter


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--meta", required=True)
    args = ap.parse_args()

    corpus = Path(args.corpus)
    con = sqlite3.connect(args.meta)
    con.execute("DROP TABLE IF EXISTS docs_fts")
    con.execute("CREATE VIRTUAL TABLE docs_fts USING fts5(path, title, body)")
    n = 0
    for p in sorted(corpus.rglob("*.md")):
        if p.name == "index.md" or "--as-of-" in p.name:
            continue
        rel = str(p.relative_to(corpus))
        fm, body = parse_front_matter(p.read_text())
        con.execute("INSERT INTO docs_fts (path, title, body) VALUES (?,?,?)",
                    (rel, fm.get("title", ""), body.strip()))
        n += 1
    con.commit()
    cnt = con.execute("SELECT COUNT(*) FROM docs_fts").fetchone()[0]
    print(f"indexed {cnt} docs (walked {n})")
    # sanity: exact citation should hit
    r = con.execute(
        "SELECT path FROM docs_fts WHERE docs_fts MATCH '7 OR cfr OR 273 OR 9' "
        "ORDER BY bm25(docs_fts, 3.0, 1.0) LIMIT 3").fetchall()
    print("sanity '7 CFR 273.9':", [x[0] for x in r])
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
