"""Retrieval tiers for the ecfr-mcp MCP server.

Tier resolution (no flags, no config files):
  ECFR_MCP_TIER=free|best  -> pinned tier (override)
  OPENROUTER_API_KEY set   -> "best"  (LLM rerank over dense candidates)
  otherwise                -> "free"  (citation fast-path + dense vectors)

"best" degrades gracefully: any LLM failure falls back to free-tier
ranking for that query rather than erroring.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# Sandbox quirk: no_proxy contains bracketed IPv6 literals that crash
# httpx/huggingface_hub at import ("Invalid port"). Sanitize first.
for _var in ("no_proxy", "NO_PROXY"):
    _val = os.environ.get(_var)
    if _val:
        os.environ[_var] = re.sub(r"\[([0-9a-fA-F:]+)\]", r"\1", _val)

import numpy as np

CITATION_RE = re.compile(r"(\d+)\s+CFR\s+([\w.]+)", re.IGNORECASE)
RERANK_MODEL = "google/gemini-2.5-flash-lite"
RERANK_API = "https://openrouter.ai/api/v1/chat/completions"
TOP_N = 50
BATCH = 10

RERANK_PROMPT = (
    "You are scoring passages for a legal researcher. Query: {q}\n\n"
    "Passages:\n{psgs}\n\n"
    "Rate each passage's relevance to the query from 0 (irrelevant) to 10 "
    "(directly answers it). Reply with ONLY a JSON object mapping passage "
    "number to score, e.g. {{\"1\": 8, \"2\": 3}}. No other text."
)


def resolve_tier(env=None) -> tuple[str, str]:
    """Return (tier, reason). tier in {"free", "best"}."""
    e = env if env is not None else os.environ
    override = (e.get("ECFR_MCP_TIER") or "").strip().lower()
    if override in ("free", "best"):
        return override, "ECFR_MCP_TIER override"
    if e.get("OPENROUTER_API_KEY"):
        return "best", "OPENROUTER_API_KEY is set"
    return "free", "no OPENROUTER_API_KEY in environment"


@dataclass
class Hit:
    citation: str
    title: str
    path: str
    program: str
    score: float
    rank: int


@dataclass
class Corpus:
    data_dir: Path
    meta: sqlite3.Connection = field(init=False)
    sections: list = field(init=False)
    sec_vecs: np.ndarray = field(init=False)
    chunk_sections: list = field(init=False)
    chunk_vecs: np.ndarray = field(init=False)
    chunk_texts: list = field(init=False)
    _embedder = None

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        d = self.data_dir
        self.meta = sqlite3.connect(d / "meta.sqlite")
        self.meta.row_factory = sqlite3.Row
        z = np.load(d / "vectors.npz")
        self.sections = z["paths"].tolist()
        V = z["vectors"].astype(np.float64)
        self.sec_vecs = V / np.linalg.norm(V, axis=1, keepdims=True)
        zc = np.load(d / "chunks.npz")
        self.chunk_sections = zc["sections"].tolist()
        Vc = zc["vectors"].astype(np.float64)
        self.chunk_vecs = Vc / np.linalg.norm(Vc, axis=1, keepdims=True)
        zt = np.load(d / "chunk_texts.npz", allow_pickle=True)
        assert zt["chunk_ids"].tolist() == zc["chunk_ids"].tolist()
        self.chunk_texts = zt["texts"].tolist()

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        return self._embedder

    def embed(self, text: str) -> np.ndarray:
        return self.embedder.encode([text], normalize_embeddings=True)[0]

    def doc_row(self, path: str):
        return self.meta.execute("SELECT * FROM docs WHERE path=?", (path,)).fetchone()

    def lookup_citation(self, citation: str):
        norm = " ".join(citation.split())
        return self.meta.execute(
            "SELECT * FROM docs WHERE citation=? AND is_current=1", (norm,)
        ).fetchone()


def _hit_from_row(row, score: float, rank: int) -> Hit:
    return Hit(citation=row["citation"], title=row["title"], path=row["path"],
               program=row["program"] or "", score=score, rank=rank)


def search_free(corpus: Corpus, query: str, top_k: int = 5) -> list[Hit]:
    """Citation fast-path + dense cosine over section vectors."""
    m = CITATION_RE.fullmatch(query.strip())
    if m:
        row = corpus.lookup_citation(f"{m.group(1)} CFR {m.group(2)}")
        if row:
            return [_hit_from_row(row, 1.0, 1)]
    q = corpus.embed(query)
    sims = corpus.sec_vecs @ q
    order = np.argsort(-sims)[:top_k]
    hits = []
    for rank, i in enumerate(order, 1):
        row = corpus.doc_row(corpus.sections[i])
        if row:
            hits.append(_hit_from_row(row, float(sims[i]), rank))
    return hits


def _llm_scores(pairs: list[tuple[str, str]], api_key: str,
                call_api=None) -> list[dict]:
    """Score batches of (query, passage) listwise. Returns list of {num: score}."""
    if call_api is None:
        def call_api(prompt: str) -> tuple[str, dict]:
            body = json.dumps({"model": RERANK_MODEL,
                               "messages": [{"role": "user", "content": prompt}],
                               "temperature": 0.0,
                               "max_tokens": 300}).encode()
            req = urllib.request.Request(
                RERANK_API, data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {api_key}",
                         "HTTP-Referer": "https://github.com/ctrimm/ecfr-mcp",
                         "X-Title": "ecfr-mcp server"})
            with urllib.request.urlopen(req, timeout=120) as r:
                payload = json.loads(r.read().decode())
            msg = payload["choices"][0]["message"].get("content") or ""
            mm = re.search(r"\{.*\}", msg, re.DOTALL)
            if not mm:
                raise RuntimeError("no JSON scores in model response")
            return mm.group(0), payload.get("usage", {})
    out = []
    for b in range(0, len(pairs), BATCH):
        chunk = pairs[b:b + BATCH]
        q = chunk[0][0]
        psgs = "\n".join(f"[{i + 1}] {p[:1500]}" for i, (_, p) in enumerate(chunk))
        raw, _ = call_api(RERANK_PROMPT.format(q=q, psgs=psgs))
        out.append(json.loads(raw))
    return out


def search_best(corpus: Corpus, query: str, top_k: int = 5,
                api_key: str | None = None,
                call_api=None) -> tuple[list[Hit], bool]:
    """LLM rerank over top-50 chunk candidates.

    Returns (hits, degraded): degraded=True means the LLM path failed and
    results are free-tier ranking instead.
    """
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    m = CITATION_RE.fullmatch(query.strip())
    if m:
        row = corpus.lookup_citation(f"{m.group(1)} CFR {m.group(2)}")
        if row:
            return [_hit_from_row(row, 1.0, 1)], False
    try:
        if not key:
            raise RuntimeError("no API key for best tier")
        q = corpus.embed(query)
        order = np.argsort(-(corpus.chunk_vecs @ q))[:TOP_N]
        pairs = [(query, corpus.chunk_texts[i]) for i in order]
        batch_scores = _llm_scores(pairs, key, call_api=call_api)
        best: dict[str, float] = {}
        for b_idx, scores in enumerate(batch_scores):
            for i in range(BATCH):
                j = b_idx * BATCH + i
                if j >= len(order):
                    break
                s = scores.get(str(i + 1), scores.get(i + 1, 0))
                try:
                    s = float(s)
                except (TypeError, ValueError):
                    s = 0.0
                sec = corpus.chunk_sections[order[j]]
                if sec not in best or s > best[sec]:
                    best[sec] = s
        ranked = sorted(best, key=lambda s: -best[s])[:top_k]
        hits = []
        for rank, sec_path in enumerate(ranked, 1):
            row = corpus.doc_row(sec_path)
            if row:
                hits.append(_hit_from_row(row, best[sec_path], rank))
        return hits, False
    except Exception:
        return search_free(corpus, query, top_k), True


def _ensure_corpus(data_dir: Path) -> Path:
    """data/corpus ships as corpus.tar.gz (thousands of files would make
    the git push crawl); extract once, on first use."""
    data_dir = Path(data_dir)
    cdir = data_dir / "corpus"
    lock = data_dir / ".corpus-extract.lock"
    if not cdir.is_dir() or not any(cdir.iterdir()):
        import fcntl
        import tarfile
        tgz = data_dir / "corpus.tar.gz"
        if not tgz.exists():
            raise FileNotFoundError(f"no corpus dir and no {tgz}")
        with open(lock, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                if not cdir.is_dir() or not any(cdir.iterdir()):
                    cdir.mkdir(parents=True, exist_ok=True)
                    with tarfile.open(tgz, "r:gz") as tf:
                        tf.extractall(data_dir)
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)
    return cdir


def get_section(data_dir: str | Path, citation: str,
                as_of: str | None = None) -> dict:
    """Full section text; as_of=YYYY-MM-DD selects the version in force then."""
    data_dir = Path(data_dir)
    _ensure_corpus(data_dir)
    con = sqlite3.connect(data_dir / "meta.sqlite")
    con.row_factory = sqlite3.Row
    norm = " ".join(citation.split())
    if as_of:
        row = con.execute(
            """SELECT * FROM docs WHERE citation=?
               AND effective_from <= ? AND (effective_to = '' OR effective_to > ?)
               ORDER BY effective_from DESC LIMIT 1""",
            (norm, as_of, as_of)).fetchone()
    else:
        row = con.execute(
            "SELECT * FROM docs WHERE citation=? AND is_current=1", (norm,)).fetchone()
    if not row:
        return {"found": False, "citation": norm}
    text = (data_dir / "corpus" / row["path"]).read_text()
    return {"found": True, "citation": row["citation"], "title": row["title"],
            "program": row["program"], "effective_from": row["effective_from"],
            "effective_to": row["effective_to"], "is_current": bool(row["is_current"]),
            "path": row["path"], "text": text}


def version_history(data_dir: str | Path, citation: str) -> list[dict]:
    data_dir = Path(data_dir)
    con = sqlite3.connect(data_dir / "meta.sqlite")
    con.row_factory = sqlite3.Row
    norm = " ".join(citation.split())
    rows = con.execute(
        "SELECT * FROM docs WHERE citation=? ORDER BY effective_from", (norm,)).fetchall()
    return [{"path": r["path"], "effective_from": r["effective_from"],
             "effective_to": r["effective_to"], "is_current": bool(r["is_current"]),
             "sha": r["sha"]} for r in rows]


def list_programs(data_dir: str | Path) -> list[dict]:
    data_dir = Path(data_dir)
    con = sqlite3.connect(data_dir / "meta.sqlite")
    rows = con.execute(
        """SELECT program, COUNT(*) n FROM docs
           WHERE is_current=1 AND program != '' GROUP BY program ORDER BY n DESC"""
    ).fetchall()
    return [{"program": p, "current_sections": n} for p, n in rows]
