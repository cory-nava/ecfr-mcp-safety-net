"""Generate a silver eval set from the corpus via OpenRouter.

Sampling strategy: 137 sections across all 15 programs, evenly spaced by
text length within each program, excluding sections already in the golden
set. Generates 2 queries per section (natural + paraphrase) — no exact
citation queries since the citation fast-path is already exhaustively tested
by the golden set.

The paraphrase type targets the hardest failure mode: the golden set scores
0.417 r@1 on paraphrase vs 0.76+ on natural, so the silver set skews toward
colloquial queries that avoid regulatory terminology.

Usage:
    OPENROUTER_API_KEY=... python silver/generate.py
    OPENROUTER_API_KEY=... python silver/generate.py --dry-run   # print 3 prompts, no API calls
    OPENROUTER_API_KEY=... python silver/generate.py --limit 5   # first 5 sections only
    OPENROUTER_API_KEY=... python silver/generate.py --model google/gemini-2.5-flash-lite

Saves incrementally to silver/eval.jsonl (resumable). Converts to
silver/eval.json on completion or when --finalize is passed.

Requires: OPENROUTER_API_KEY (no extra pip installs — uses stdlib urllib).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = ROOT / "golden" / "eval.json"
DATA_DIR = ROOT / "data"
DEFAULT_JSONL = ROOT / "silver" / "eval.jsonl"
DEFAULT_OUT = ROOT / "silver" / "eval.json"

OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-flash-lite"

SYSTEM = (
    "You are creating evaluation questions for a regulatory text retrieval benchmark. "
    "A retrieval system gets your question and must return this exact CFR section — "
    "not just a related one — so make questions that are specific enough to point to "
    "this section but that don't just quote it verbatim.\n\n"
    "Rules:\n"
    "- Each question must be answerable specifically from THIS section.\n"
    "- The NATURAL question: professional tone, may use regulatory concepts, "
    "  must NOT include the citation (e.g. '7 CFR 273.9').\n"
    "- The PARAPHRASE question: plain everyday language a benefits recipient or "
    "  layperson would use. MUST NOT use regulatory terms, acronyms, or "
    "  program-specific jargon from the section text. Rephrase the core concept "
    "  in ordinary words. Must be clearly different from the natural question.\n\n"
    "Return ONLY a JSON array, no other text:\n"
    '[{"type": "natural", "query": "..."}, {"type": "paraphrase", "query": "..."}]'
)

# Sections to sample per program. Balanced to give full coverage of programs
# absent from or underrepresented in the golden set while keeping the total
# to ~137 sections (274 queries).
TARGETS: dict[str, int] = {
    "ssi": 20,
    "medicaid": 20,
    "lifeline": 12,
    "housing": 12,
    "medicare": 10,
    "snap": 10,
    "tanf": 8,
    "school-meals": 8,
    "liheap": 8,
    "tefap": 6,
    "ccdf": 6,
    "summer-meals": 5,
    "wic": 4,
    "cacfp": 4,
    "unemployment-insurance": 4,
}


def _extract_body(md: str, max_chars: int = 1500) -> str:
    """Strip YAML front matter, return body up to max_chars."""
    lines = md.split("\n")
    start = 0
    if lines and lines[0].strip() == "---":
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                start = i + 1
                break
    return "\n".join(lines[start:]).strip()[:max_chars]


def _load_corpus_texts() -> dict[str, str]:
    out: dict[str, str] = {}
    with tarfile.open(DATA_DIR / "corpus.tar.gz", "r:gz") as tf:
        for m in tf:
            if m.isfile() and m.name.startswith("corpus/") and m.name.endswith(".md"):
                out[m.name[len("corpus/"):]] = tf.extractfile(m).read().decode("utf-8")
    return out


def _sample_sections(golden_paths: set[str]) -> list[dict]:
    """Return list of {path, citation, title, program} dicts."""
    con = sqlite3.connect(DATA_DIR / "meta.sqlite")
    con.row_factory = sqlite3.Row
    texts_len: dict[str, int] = {}
    with tarfile.open(DATA_DIR / "corpus.tar.gz", "r:gz") as tf:
        for m in tf:
            if m.isfile() and m.name.startswith("corpus/") and m.name.endswith(".md"):
                texts_len[m.name[len("corpus/"):]] = m.size

    selected = []
    for prog, n in TARGETS.items():
        rows = con.execute(
            "SELECT path, citation, title FROM docs WHERE is_current=1 AND program=?",
            (prog,),
        ).fetchall()
        candidates = [
            (r["path"], r["citation"], r["title"], texts_len.get(r["path"], 0))
            for r in rows
            if r["path"] not in golden_paths and texts_len.get(r["path"], 0) >= 100
        ]
        candidates.sort(key=lambda x: -x[3])
        step = max(1, len(candidates) // n)
        for path, citation, title, _ in candidates[::step][:n]:
            selected.append({"path": path, "citation": citation, "title": title, "program": prog})
    return selected


def _ask(title: str, citation: str, body: str, api_key: str, model: str) -> list[dict]:
    user = f"Citation: {citation}\nTitle: {title}\n\nText excerpt:\n{body}"
    payload = json.dumps({
        "model": model,
        "max_tokens": 512,
        "temperature": 0.3,
        "messages": [
            {"role": "user", "content": SYSTEM + "\n\n" + user},
        ],
    }).encode()
    req = urllib.request.Request(
        OPENROUTER_API,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/ctrimm/ecfr-mcp",
            "X-Title": "ecfr-mcp silver-eval-gen",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read().decode())
    text = (resp["choices"][0]["message"].get("content") or "").strip()
    mm = re.search(r"\[.*\]", text, re.DOTALL)
    if not mm:
        raise RuntimeError(f"no JSON array in response: {text[:200]!r}")
    return json.loads(mm.group(0))


def _finalize(jsonl_path: Path, out_path: Path) -> int:
    records = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    queries = []
    qid = 1
    for rec in records:
        for q in rec["queries"]:
            queries.append({
                "id": qid,
                "query": q["query"],
                "expected_citation": rec["citation"],
                "expected_path": rec["path"],
                "type": q["type"],
                "program": rec["program"],
            })
            qid += 1
    out_path.write_text(json.dumps(queries, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(queries)} queries to {out_path}")
    return len(queries)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--jsonl", default=str(DEFAULT_JSONL))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"OpenRouter model for generation (default: {DEFAULT_MODEL})")
    ap.add_argument("--finalize", action="store_true", help="convert existing JSONL to JSON without generating")
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    out_path = Path(args.out)

    if args.finalize:
        _finalize(jsonl_path, out_path)
        return 0

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key and not args.dry_run:
        sys.exit("OPENROUTER_API_KEY is not set")

    golden_paths = {q["expected_path"] for q in json.loads(GOLDEN_PATH.read_text())}
    sections = _sample_sections(golden_paths)
    print(f"Sampled {len(sections)} sections across {len(TARGETS)} programs | model={args.model}", flush=True)

    done_paths: set[str] = set()
    if jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            if line.strip():
                done_paths.add(json.loads(line)["path"])
        print(f"Resuming: {len(done_paths)} already done", flush=True)

    corpus_texts = _load_corpus_texts()
    to_do = [s for s in sections if s["path"] not in done_paths]
    if args.limit:
        to_do = to_do[: args.limit]

    fh = open(jsonl_path, "a", encoding="utf-8")
    success = 0
    for i, sec in enumerate(to_do):
        body = _extract_body(corpus_texts.get(sec["path"], ""))
        if not body:
            print(f"  skip {sec['citation']} (empty body)", flush=True)
            continue

        if args.dry_run:
            if i < 3:
                print(f"\n--- DRY RUN: {sec['citation']} ---")
                print(f"Title: {sec['title']}")
                print(f"Body excerpt: {body[:200]!r}")
            continue

        for attempt in range(3):
            try:
                queries = _ask(sec["title"], sec["citation"], body, api_key, args.model)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  FAILED {sec['citation']}: {e}", flush=True)
                    queries = None
                    break
                time.sleep(5 * (attempt + 1))

        if queries is None:
            continue

        rec = {"path": sec["path"], "citation": sec["citation"],
               "program": sec["program"], "queries": queries}
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        success += 1
        print(f"  [{success}/{len(to_do)}] {sec['citation']} → {[q['type'] for q in queries]}", flush=True)
        time.sleep(0.3)

    fh.close()

    if not args.dry_run:
        n = _finalize(jsonl_path, out_path)
        print(f"\nDone. {success} new sections generated, {n} total queries in {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
