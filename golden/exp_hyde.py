"""Experiment: HyDE (Hypothetical Document Embeddings).

Step 1 (generate): for each golden query, ask an LLM to write a short
passage in CFR style that would answer it. Saves golden/hyde_passages.json.
Step 2 (eval): embed the HyDE passages (plain, no prefix) and retrieve
against section vectors; also test query+HyDE vector averaging.
Citation fast-path applies as in production.

OpenRouter auth via Secure Vault surrogate (custom.openrouter).
Cost: ~50 queries x ~400 completion tokens on a flash-class model (~$0.01).

Usage:
    python3 golden/exp_hyde.py --step generate [--limit 4]   # pilot with --limit
    python3 golden/exp_hyde.py --step eval --vectors build/vectors.npz --meta build/meta.sqlite
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

CITATION_RE = re.compile(r"(\d+)\s+CFR\s+([\w.]+)", re.IGNORECASE)
MODEL = "deepseek/deepseek-v4-flash"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
PROMPT = ("Write a short passage (4-6 sentences) in the style of a U.S. federal "
          "benefits regulation (Code of Federal Regulations) that answers the "
          "following question. Use formal regulatory terminology. Write only "
          "the passage, no disclaimers.\n\nQuestion: {q}")


def ask_or(model: str, prompt: str) -> tuple[str, dict]:
    sys.path.insert(0, "/opt/hatch/skills/skill-creator/bin")
    from dynamic_credentials import add_surrogate_to_request
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.7, "max_tokens": 400}).encode()
    req = urllib.request.Request(API_URL, data=body,
                                 headers={"Content-Type": "application/json",
                                          "HTTP-Referer": "https://github.com/ctrimm/ecfr-mcp",
                                          "X-Title": "ecfr-mcp hyde experiment"})
    add_surrogate_to_request(req, "custom.openrouter", allowed_hosts=["openrouter.ai"])
    import time
    last = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                payload = json.loads(r.read().decode())
        except Exception as e:
            last = e
            time.sleep(10 * (attempt + 1))
            continue
        last = payload
        msg = payload["choices"][0]["message"].get("content")
        if msg:
            return msg.strip(), payload.get("usage", {})
        time.sleep(5)
    raise RuntimeError(f"no usable response after retries: {str(last)[:300]}")


def do_generate(args) -> int:
    goldens = json.load(open(args.golden))[:args.limit] if args.limit else json.load(open(args.golden))
    out_path = Path(args.out)
    existing = json.load(open(out_path)) if out_path.exists() else []
    done_ids = {e["id"] for e in existing}
    total_in = total_out = 0
    failed = []
    for g in goldens:
        if g["id"] in done_ids:
            continue
        try:
            passage, usage = ask_or(args.model, PROMPT.format(q=g["query"]))
        except Exception as e:
            print(f"  FAILED {g['id']}: {type(e).__name__} {str(e)[:100]}", flush=True)
            failed.append(g["id"])
            continue
        total_in += usage.get("prompt_tokens", 0)
        total_out += usage.get("completion_tokens", 0)
        existing.append({"id": g["id"], "query": g["query"], "passage": passage,
                         "model": args.model, "usage": usage})
        out_path.write_text(json.dumps(existing, indent=1))
        done_ids.add(g["id"])
        print(f"  generated {g['id']} ({len(passage)} chars)", flush=True)
        time.sleep(2)
    print(f"pilot/full usage: {total_in} prompt + {total_out} completion tokens")
    print(f"saved {len(existing)} passages -> {out_path}; failed: {failed}")
    return 0 if not failed else 2


def do_eval(args) -> int:
    from sentence_transformers import SentenceTransformer
    vec = np.load(args.vectors)
    V, paths = vec["vectors"], vec["paths"].tolist()
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    con = sqlite3.connect(args.meta)
    model = SentenceTransformer(args.model_emb)
    passages = {e["id"]: e["passage"] for e in json.load(open(args.passages))}
    goldens = json.load(open(args.golden))

    def ranks_for(qvec):
        sims = Vn @ qvec
        return list(np.array(paths)[np.argsort(-sims)])

    for name in ("hyde_only", "hyde_plus_query"):
        results = []
        for g in goldens:
            hp = model.encode([passages[g["id"]]], normalize_embeddings=True)[0]
            if name == "hyde_plus_query":
                qq = model.encode([g["query"]], normalize_embeddings=True)[0]
                qv = hp + qq
                qv = qv / np.linalg.norm(qv)
            else:
                qv = hp
            ranked = ranks_for(qv)
            m = CITATION_RE.fullmatch(g["query"].strip())
            if m:
                hit = con.execute("SELECT path FROM docs WHERE citation=? AND is_current=1",
                                  (f"{m.group(1)} CFR {m.group(2)}",)).fetchone()
                if hit and hit[0] in ranked:
                    ranked.remove(hit[0])
                    ranked.insert(0, hit[0])
            want = g["expected_path"]
            rank = ranked.index(want) + 1 if want in ranked else 10**9
            results.append({"type": g["type"], "rank": rank})
        n = len(results)
        line = f"{name:16s} " + " ".join(
            f"r@{k}={sum(1 for r in results if r['rank'] <= k) / n:.3f}" for k in (1, 3, 5))
        print(line)
        for t in ("exact", "natural", "paraphrase"):
            sub = [r for r in results if r["type"] == t]
            print(f"  {t:10s} " + " ".join(
                f"r@{k}={sum(1 for r in sub if r['rank'] <= k) / len(sub):.3f}" for k in (1, 3, 5)))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["generate", "eval"], required=True)
    ap.add_argument("--limit", type=int, default=0, help="pilot: only first N queries")
    ap.add_argument("--golden", default=str(Path(__file__).parent / "eval.json"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "hyde_passages.json"))
    ap.add_argument("--passages", default=str(Path(__file__).parent / "hyde_passages.json"))
    ap.add_argument("--vectors", default="")
    ap.add_argument("--meta", default="")
    ap.add_argument("--model", default=MODEL, help="OpenRouter model for generation")
    ap.add_argument("--model-emb", default="BAAI/bge-base-en-v1.5")
    args = ap.parse_args()
    if args.step == "generate":
        return do_generate(args)
    if not args.vectors or not args.meta:
        ap.error("--step eval needs --vectors and --meta")
    return do_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
