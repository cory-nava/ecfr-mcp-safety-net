"""Pack the ecfr-markdown corpus into data/corpus.tar.gz, which the server reads section text from.

Deterministic (sorted members, fixed mtime and owner), so the tarball only changes when the corpus does.

Usage:
    python build/pack_corpus.py --corpus ../ecfr-markdown-safety-net/corpus --out data
"""
from __future__ import annotations

import argparse
import gzip
import io
import tarfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    corpus = Path(args.corpus)
    files = sorted(p for p in corpus.rglob("*") if p.is_file() and p.suffix in (".md", ".json"))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for p in files:
            data = p.read_bytes()
            info = tarfile.TarInfo(f"corpus/{p.relative_to(corpus).as_posix()}")
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tf.addfile(info, io.BytesIO(data))
    out = Path(args.out) / "corpus.tar.gz"
    with open(out, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0, filename="") as gz:
        gz.write(buf.getvalue())
    print(f"packed {len(files)} files -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
