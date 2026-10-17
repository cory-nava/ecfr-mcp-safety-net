"""Experiment: LLM-as-reranker over chunk candidates (OpenRouter) — candidate variants.

Variant of exp_llm_rerank.py: --top-n sets the dense candidate depth,
--candidate-source hybrid unions dense top-N with BM25 top-N chunk candidates.

Step 1 (generate): for each golden query, take top-50 chunks by vector
similarity and have an OpenRouter chat model score them listwise
(10 chunks per call -> 5 calls/query -> 250 calls). Scores are saved
incrementally to a local JSONL: golden/llm_rerank_scores.jsonl.
Step 2 (eval): pure-local scoring from the JSONL (no API): section rank =
best chunk rank. Citation fast-path applies.

Cost guard: cumulative spend tracked from usage.cost; aborts over $1.00.
Expected total: ~$0.03 on deepseek-v4-flash.

Usage:
    python3 golden/exp_llm_rerank.py --step generate [--limit 4]
    python3 golden/exp_llm_rerank.py --step eval
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "build"))
from build_chunks import chunk_tokens, CHUNK_TOKENS, OVERLAP
from build_vectors import parse_front_matter

CITATION_RE = re.compile(r"(\d+)\s+CFR\s+([\w.]+)", re.IGNORECASE)
MODEL = "google/gemini-2.5-flash-lite"  # default; override with --model-llm
API_URL = "https://openrouter.ai/api/v1/chat/completions"
BUDGET = 1.00
BATCH = 10
TOP_N = 50

PROMPT = ("You are scoring passages for a legal researcher. Query: {q}\n\n"
          "Passages:\n{psgs}\n\n"
          "Rate each passage's relevance to the query from 0 (irrelevant) to 10 "
          "(directly answers it). Reply with ONLY a JSON object mapping passage "
          "number to score, e.g. {{\"1\": 8, \"2\": 3}}. No other text.")



def _curl_post(url: str, body: bytes, headers: dict) -> dict:
    """POST via curl subprocess. Python's urllib intermittently hangs on some
    OpenRouter provider routes from this sandbox; curl is reliable."""
    import subprocess as _sp
    cmd = ["curl", "-sS", "-m", "90", "-X", "POST", url,
           "-H", "Expect:"]  # disable 100-continue: hangs on large bodies here
    for k, v in headers.items():
        cmd += ["-H", f"{k}: {v}"]
    cmd += ["--data-binary", "@-", "-w", "\n%{http_code}"]
    p = _sp.run(cmd, input=body, capture_output=True, timeout=120)
    if p.returncode != 0:
        raise RuntimeError(f"curl rc={p.returncode}: {p.stderr[:160]!r}")
    out = p.stdout.decode().rsplit("\n", 1)
    code = int(out[1]) if len(out) == 2 and out[1].strip().isdigit() else 0
    payload = json.loads(out[0]) if out[0].strip() else {}
    if code in (401, 403):
        raise urllib.error.HTTPError(url, code, "forbidden", {}, None)
    if code != 200:
        raise RuntimeError(f"HTTP {code}: {str(out[0])[:160]}")
    return payload


def ask_or(prompt: str, model: str = MODEL) -> tuple[dict, dict]:
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.0, "max_tokens": 300}).encode()
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    last = None
    timeouts = 0
    for attempt in range(12):
        try:
            payload = _curl_post(API_URL, body, headers)
        except urllib.error.HTTPError as e:
            # 403s here are the transient edge block (isolated identical
            # requests succeed): back off and retry. 401 = bad credential,
            # fail fast.
            if e.code == 401:
                raise RuntimeError("HTTP 401 (not retrying)")
            last = e
            if attempt == 0:
                print(f"  ask_or attempt 0 failed: HTTPError {e.code}",
                      flush=True)
            time.sleep(15 * (attempt + 1))
            continue
        except Exception as e:
            last = e
            if attempt == 0:
                print(f"  ask_or attempt 0 failed: {type(e).__name__}: {str(e)[:200]}",
                      flush=True)
            if "timed out" in type(e).__name__.lower() or "timeout" in str(e).lower():
                timeouts += 1
                if timeouts >= 3:
                    raise RuntimeError(f"3 consecutive call timeouts: {str(e)[:200]}")
            time.sleep(15 * (attempt + 1))
            continue
        last = payload
        msg = payload["choices"][0]["message"].get("content")
        if msg:
            m = re.search(r"\{.*\}", msg, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0)), payload.get("usage", {})
                except json.JSONDecodeError:
                    pass
        time.sleep(10)
    raise RuntimeError(f"no usable scores after retries: {str(last)[:300]}")


def _fts_query(q: str) -> str:
    toks = re.findall(r"[a-z0-9]+", q.lower())
    return " OR ".join(toks[:20]) if toks else '""'


def get_bm25(db_path: str, texts: list[str]):
    """FTS5 BM25 over chunk texts; builds the index on first use."""
    import sqlite3 as _sq
    con = _sq.connect(db_path)
    n = con.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE name='chunks_fts'").fetchone()[0]
    if n == 0:
        print("building BM25 chunk index...", flush=True)
        con.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(text)")
        con.executemany("INSERT INTO chunks_fts(text) VALUES (?)",
                        [(t,) for t in texts])
        con.commit()
        print(f"  indexed {len(texts)} chunks", flush=True)
    return con


def bm25_top(con, q: str, n: int) -> list[int]:
    fq = _fts_query(q)
    if not fq.strip():
        return []
    rows = con.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
        "ORDER BY rank LIMIT ?", (fq, n)).fetchall()
    return [r[0] - 1 for r in rows]


def load_chunks(args):

    from sentence_transformers import SentenceTransformer
    z = np.load(args.chunks)
    V = z["vectors"]
    saved_ids = z["chunk_ids"].tolist()
    sections = z["sections"].tolist()
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    model = SentenceTransformer(args.model_emb)
    cache = Path(args.chunks).parent / "chunk_texts.npz"
    if cache.exists():
        c = np.load(cache, allow_pickle=True)
        assert c["chunk_ids"].tolist() == saved_ids, "text cache misaligned"
        print(f"loaded {len(c['texts'])} chunk texts from cache", flush=True)
        return model, Vn, None, sections, c["texts"].tolist()
    tok = model.tokenizer
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
    assert ids == saved_ids, "chunk regen misaligned"
    return model, Vn, z["paths"].tolist() if "paths" in z else None, sections, texts


def do_generate(args) -> int:
    model, Vn, _, sections, texts = load_chunks(args)
    out = Path(args.scores)
    done, spent = set(), 0.0
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                done.add(e["id"])
                spent += e.get("cost", 0.0)
    fh = open(out, "a")
    goldens = json.load(open(args.golden))
    if args.limit:
        goldens = goldens[:args.limit]
    hyde_vecs = {}
    if args.candidate_source == "hyde_avg":
        passages = {e["id"]: e["passage"] for e in json.load(open(args.hyde_passages))}
        for g in goldens:
            hp = model.encode([passages[g["id"]]], normalize_embeddings=True)[0]
            qq = model.encode([g["query"]], normalize_embeddings=True)[0]
            qv = hp + qq
            hyde_vecs[g["id"]] = qv / np.linalg.norm(qv)
        print("HyDE-averaged candidate vectors ready", flush=True)
    bm25_con = None
    if args.candidate_source == "hybrid":
        bm25_con = get_bm25(args.bm25_db, texts)
    # OPENROUTER_API_KEY comes from the environment.
    failed = []
    for g in goldens:
        if g["id"] in done or spent >= BUDGET:
            continue
        qv = hyde_vecs.get(g["id"], model.encode([g["query"]], normalize_embeddings=True)[0])
        order = np.argsort(-(Vn @ qv))[:args.top_n].tolist()
        if args.candidate_source == "hybrid":
            for j in bm25_top(bm25_con, g["query"], args.top_n):
                if j not in order:
                    order.append(j)
        chunk_scores: dict[str, float] = {}
        qcost = 0.0
        ok = True
        for b in range(0, len(order), BATCH):
            idxs = order[b:b + BATCH]
            psgs = "\n".join(f"[{i + 1}] {texts[j][:1500]}"
                             for i, j in enumerate(idxs))
            print(f"  q{g['id']} batch {b}: calling {args.model_llm}", flush=True)
            t0 = time.time()
            try:
                scores, usage = ask_or(PROMPT.format(q=g["query"], psgs=psgs), args.model_llm)
            except Exception as e:
                print(f"  FAILED q{g['id']} batch {b}: {type(e).__name__}: {str(e)[:250]}",
                      flush=True)
                ok = False
                break
            print(f"  q{g['id']} batch {b}: ok in {time.time()-t0:.1f}s", flush=True)
            cost = usage.get("cost", 0.0) or 0.0
            spent += cost
            qcost += cost
            for i, j in enumerate(idxs):
                s = scores.get(str(i + 1), scores.get(i + 1, 0))
                try:
                    chunk_scores[str(j)] = float(s)
                except (TypeError, ValueError):
                    chunk_scores[str(j)] = 0.0
            time.sleep(5)
            if spent >= BUDGET:
                print(f"  BUDGET REACHED: ${spent:.4f}", flush=True)
                break
        if not ok:
            failed.append(g["id"])
            continue
        rec = {"id": g["id"], "chunk_scores": chunk_scores, "cost": qcost,
               "model": args.model_llm, "candidate_source": args.candidate_source}
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        done.add(g["id"])
        print(f"  scored q{g['id']} ({len(chunk_scores)} chunks) "
              f"qcost=${qcost:.4f} spent=${spent:.4f}", flush=True)
    print(f"done: {len(done)} queries scored, spent=${spent:.4f}, failed={failed}")
    Path(args.scores + ".spent").write_text(f"{spent:.6f}")
    return 0 if not failed else 2


def do_eval(args) -> int:
    z = np.load(args.chunks)
    sections = z["sections"].tolist()
    con = sqlite3.connect(args.meta)
    goldens = {g["id"]: g for g in json.load(open(args.golden))}
    scored = {}
    for line in Path(args.scores).read_text().splitlines():
        if line.strip():
            e = json.loads(line)
            scored[e["id"]] = e["chunk_scores"]
    results = []
    for qid, cs in scored.items():
        g = goldens[qid]
        m = CITATION_RE.fullmatch(g["query"].strip())
        if m:
            hit = con.execute("SELECT path FROM docs WHERE citation=? AND is_current=1",
                              (f"{m.group(1)} CFR {m.group(2)}",)).fetchone()
            if hit:
                # citation fast-path: a query that IS a citation is a lookup
                results.append({"type": g["type"], "rank": 1})
                continue
        best: dict[str, float] = {}
        for jstr, s in cs.items():
            sec = sections[int(jstr)]
            if sec not in best or s > best[sec]:
                best[sec] = float(s)
        ranked = sorted(best, key=lambda s: -best[s])
        want = g["expected_path"]
        rank = ranked.index(want) + 1 if want in ranked else 10**9
        results.append({"type": g["type"], "rank": rank})
    n = len(results)
    print(f"n={n} (LLM listwise rerank)")
    for k in (1, 3, 5):
        print(f"recall@{k}: {sum(1 for r in results if r['rank'] <= k) / n:.3f}")
    for t in ("exact", "natural", "paraphrase"):
        sub = [r for r in results if r["type"] == t]
        print(f"  {t:10s} n={len(sub):2d}  " +
              "  ".join(f"r@{k}={sum(1 for r in sub if r['rank'] <= k) / len(sub):.3f}"
                        for k in (1, 3, 5)))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["generate", "eval"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--golden", default=str(Path(__file__).parent / "eval.json"))
    ap.add_argument("--chunks", default="build/chunks.npz")
    ap.add_argument("--meta", default="build/meta.sqlite")
    ap.add_argument("--corpus", default="corpus-src/corpus")
    ap.add_argument("--scores", default=str(Path(__file__).parent / "llm_rerank_scores.jsonl"))
    ap.add_argument("--model-emb", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--candidate-source", choices=["query", "hyde_avg", "hybrid"], default="query",
                    help="hyde_avg: chunks by HyDE-passage+query averaged vector; hybrid: dense top-N union BM25 top-N")
    ap.add_argument("--top-n", type=int, default=50,
                    help="dense candidate depth (hybrid adds up to top-N BM25 extras)")
    ap.add_argument("--bm25-db", default="build/bm25_chunks.sqlite")
    ap.add_argument("--model-llm", default=MODEL)
    ap.add_argument("--hyde-passages", default=str(Path(__file__).parent / "hyde_passages.json"))
    args = ap.parse_args()
    if args.step == "generate":
        return do_generate(args)
    return do_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
