"""Experiment: LLM-as-reranker over chunk candidates (OpenRouter).

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
MODEL = "google/gemini-2.5-flash-lite"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
BUDGET = 1.00
BATCH = 10
TOP_N = 50

PROMPT = ("You are scoring passages for a legal researcher. Query: {q}\n\n"
          "Passages:\n{psgs}\n\n"
          "Rate each passage's relevance to the query from 0 (irrelevant) to 10 "
          "(directly answers it). Reply with ONLY a JSON object mapping passage "
          "number to score, e.g. {{\"1\": 8, \"2\": 3}}. No other text.")


def ask_or(prompt: str) -> tuple[dict, dict]:
    sys.path.insert(0, "/opt/hatch/skills/skill-creator/bin")
    from dynamic_credentials import add_surrogate_to_request
    body = json.dumps({"model": MODEL,
                       "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.0, "max_tokens": 300}).encode()
    req = urllib.request.Request(API_URL, data=body,
                                 headers={"Content-Type": "application/json",
                                          "HTTP-Referer": "https://github.com/ctrimm/ecfr-mcp",
                                          "X-Title": "ecfr-mcp llm-rerank"})
    add_surrogate_to_request(req, "custom.openrouter", allowed_hosts=["openrouter.ai"])
    last = None
    for attempt in range(12):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                payload = json.loads(r.read().decode())
        except Exception as e:
            last = e
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
    failed = []
    for g in goldens:
        if g["id"] in done or spent >= BUDGET:
            continue
        qv = hyde_vecs.get(g["id"], model.encode([g["query"]], normalize_embeddings=True)[0])
        order = np.argsort(-(Vn @ qv))[:TOP_N]
        chunk_scores: dict[str, float] = {}
        qcost = 0.0
        ok = True
        for b in range(0, TOP_N, BATCH):
            idxs = order[b:b + BATCH]
            psgs = "\n".join(f"[{i + 1}] {texts[j][:1500]}"
                             for i, j in enumerate(idxs))
            try:
                scores, usage = ask_or(PROMPT.format(q=g["query"], psgs=psgs))
            except Exception as e:
                print(f"  FAILED q{g['id']} batch {b}: {type(e).__name__}: {str(e)[:250]}",
                      flush=True)
                ok = False
                break
            cost = usage.get("cost", 0.0) or 0.0
            spent += cost
            qcost += cost
            for i, j in enumerate(idxs):
                s = scores.get(str(i + 1), scores.get(i + 1, 0))
                try:
                    chunk_scores[str(j)] = float(s)
                except (TypeError, ValueError):
                    chunk_scores[str(j)] = 0.0
            time.sleep(2)
            if spent >= BUDGET:
                print(f"  BUDGET REACHED: ${spent:.4f}", flush=True)
                break
        if not ok:
            failed.append(g["id"])
            continue
        rec = {"id": g["id"], "chunk_scores": chunk_scores, "cost": qcost,
               "model": MODEL, "candidate_source": args.candidate_source}
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
    print(f"n={n} (LLM listwise rerank top-{TOP_N} chunks, {MODEL})")
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
    ap.add_argument("--candidate-source", choices=["query", "hyde_avg"], default="query",
                    help="hyde_avg: top-50 chunks by HyDE-passage+query averaged vector")
    ap.add_argument("--hyde-passages", default=str(Path(__file__).parent / "hyde_passages.json"))
    args = ap.parse_args()
    if args.step == "generate":
        return do_generate(args)
    return do_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
