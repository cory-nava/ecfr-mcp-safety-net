"""Retrieval tiers for the ecfr-mcp MCP server.

Tier resolution (no flags, no config files):
  ECFR_MCP_TIER=free|best  -> pinned tier (override)
  OPENROUTER_API_KEY set   -> "best"  (LLM rerank over dense candidates)
  otherwise                -> "free"  (citation fast-path + dense vectors)

"best" degrades gracefully: any LLM failure falls back to free-tier
ranking for that query rather than erroring.
"""
from __future__ import annotations

import functools
import json
import os
import re
import sqlite3
import tarfile
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

CITATION_RE = re.compile(r"(\d+)\s+CFR\s+([\w.]+)", re.IGNORECASE)
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
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


@functools.lru_cache(maxsize=None)
def _db(data_dir: Path) -> sqlite3.Connection:
    """One read-only connection per data dir (a missing file errors instead of creating an empty DB)."""
    path = Path(data_dir) / "meta.sqlite"
    if not path.exists():
        raise FileNotFoundError(path)
    con = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


@functools.lru_cache(maxsize=None)
def _texts(data_dir: Path) -> dict[str, str]:
    """Corpus Markdown keyed by path. Read once from corpus.tar.gz into memory (~30MB),
    so nothing is ever written next to the package."""
    out = {}
    with tarfile.open(Path(data_dir) / "corpus.tar.gz", "r:gz") as tf:
        for m in tf:
            if m.isfile() and m.name.startswith("corpus/") and m.name.endswith(".md"):
                out[m.name[len("corpus/"):]] = tf.extractfile(m).read().decode("utf-8")
    return out


@dataclass
class Hit:
    citation: str
    title: str
    path: str
    program: str
    score: float
    rank: int


class Corpus:
    def __init__(self, data_dir: str | Path):
        self.data_dir = d = Path(data_dir)
        self.meta = _db(d)
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        # the query encoder must be the model that built the indexes; the manifest is the source of truth
        self.model_name = manifest.get("model", DEFAULT_EMBED_MODEL)
        z = np.load(d / "vectors.npz")
        self.sections = z["paths"].tolist()
        self.sec_vecs = _normalized(z["vectors"])
        zc = np.load(d / "chunks.npz")
        self.chunk_sections = zc["sections"].tolist()
        self.chunk_vecs = _normalized(zc["vectors"])
        zt = np.load(d / "chunk_texts.npz", allow_pickle=True)  # our own build output, shipped with the code
        if zt["chunk_ids"].tolist() != zc["chunk_ids"].tolist():
            raise ValueError("chunk_texts.npz is misaligned with chunks.npz; rebuild with build/build_chunks.py")
        self.chunk_texts = zt["texts"].tolist()
        dims = {self.sec_vecs.shape[1], self.chunk_vecs.shape[1], int(manifest.get("dims", self.sec_vecs.shape[1]))}
        if len(dims) != 1:
            raise ValueError(f"index dimensions disagree ({sorted(dims)}): vectors.npz, chunks.npz and "
                             f"manifest.json must all come from {self.model_name}; rebuild the indexes")
        self.dims = dims.pop()
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder

    def embed(self, text: str) -> np.ndarray:
        q = self.embedder.encode([text], normalize_embeddings=True)[0]
        if q.shape[0] != self.dims:
            raise ValueError(f"{self.model_name} gives {q.shape[0]}-dim vectors but the indexes are {self.dims}-dim")
        return q

    def doc_row(self, path: str):
        return self.meta.execute("SELECT * FROM docs WHERE path=?", (path,)).fetchone()

    def lookup_citation(self, citation: str):
        norm = " ".join(citation.split())
        return self.meta.execute(
            "SELECT * FROM docs WHERE citation=? AND is_current=1", (norm,)
        ).fetchone()


def _normalized(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _hit_from_row(row, score: float, rank: int) -> Hit:
    return Hit(citation=row["citation"], title=row["title"], path=row["path"],
               program=row["program"] or "", score=score, rank=rank)


def _citation_hit(corpus: Corpus, query: str) -> Hit | None:
    m = CITATION_RE.fullmatch(query.strip())
    row = m and corpus.lookup_citation(f"{m.group(1)} CFR {m.group(2)}")
    return _hit_from_row(row, 1.0, 1) if row else None


def search_free(corpus: Corpus, query: str, top_k: int = 5) -> list[Hit]:
    """Citation fast-path + dense cosine over section vectors."""
    hit = _citation_hit(corpus, query)
    if hit:
        return [hit]
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
    hit = _citation_hit(corpus, query)
    if hit:
        return [hit], False
    if not key:
        return search_free(corpus, query, top_k), True
    # embedding errors are not LLM failures: let them surface instead of hiding behind the free tier
    q = corpus.embed(query)
    try:
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


def get_section(data_dir: str | Path, citation: str,
                as_of: str | None = None) -> dict:
    """Full section text; as_of=YYYY-MM-DD selects the version in force then.

    effective_to is the last day a version was in force (inclusive), as ecfr-markdown writes it.
    """
    data_dir = Path(data_dir)
    con = _db(data_dir)
    norm = " ".join(citation.split())
    if as_of:
        try:
            as_of = _iso_date(as_of)
        except ValueError:
            # dates compare as strings below, so a malformed one would silently pick the wrong version
            return {"found": False, "citation": norm, "status": "invalid_date",
                    "note": f"as_of must be a calendar date written YYYY-MM-DD; got {as_of!r}."}
    if as_of:
        row = con.execute(
            """SELECT * FROM docs WHERE citation=?
               AND effective_from <= ? AND (effective_to = '' OR effective_to >= ?)
               ORDER BY effective_from DESC LIMIT 1""",
            (norm, as_of, as_of)).fetchone()
    else:
        row = con.execute(
            "SELECT * FROM docs WHERE citation=? AND is_current=1", (norm,)).fetchone()
    if not row:
        return _miss(norm, as_of, version_history(data_dir, norm))
    return {"found": True, "citation": row["citation"], "title": row["title"],
            "program": row["program"], "effective_from": row["effective_from"],
            "effective_to": row["effective_to"], "is_current": bool(row["is_current"]),
            "path": row["path"], "text": _texts(data_dir)[row["path"]]}


def _iso_date(s: str) -> str:
    """Strict YYYY-MM-DD, returned unchanged; anything else (2024-6-1, June 2024, a datetime) raises."""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s.strip()):
        raise ValueError(s)
    return date.fromisoformat(s.strip()).isoformat()


def _miss(norm: str, as_of: str | None, hist: list[dict]) -> dict:
    """Explain why get_section found nothing, without guessing beyond what the version history shows."""
    if not hist:
        return {"found": False, "citation": norm, "status": "unknown_citation"}
    first, last = hist[0], hist[-1]
    span = f"{len(hist)} recorded version(s), the earliest in force from {first['effective_from']}"
    if as_of is None or as_of > (last["effective_to"] or "9999-12-31"):
        # no version is current: every recorded version has an end date
        return {"found": False, "citation": norm, "status": "not_in_force",
                "versions": len(hist),
                "last_effective_from": last["effective_from"],
                "last_effective_to": last["effective_to"],
                "note": (f"{norm} has no text in force{' on ' + as_of if as_of else ' today'}: its last "
                         f"version was in force {last['effective_from']} to {last['effective_to']}. "
                         "The section was likely repealed, marked [Reserved], or renumbered; "
                         "call version_history for the full record.")}
    if as_of < first["effective_from"]:
        return {"found": False, "citation": norm, "status": "before_recorded_history",
                "versions": len(hist), "earliest_effective_from": first["effective_from"],
                "note": (f"The corpus has no text for {norm} on {as_of}: its {span}. That says nothing "
                         "about whether the section was in force then; check the official CFR for "
                         "earlier dates.")}
    ended = max((h["effective_to"] for h in hist if h["effective_to"] and h["effective_to"] < as_of), default="")
    starts = min((h["effective_from"] for h in hist if h["effective_from"] > as_of), default="")
    return {"found": False, "citation": norm, "status": "not_in_force_on_date",
            "versions": len(hist), "gap_from": ended, "gap_to": starts,
            "note": (f"No version of {norm} was in force on {as_of}"
                     + (f": the previous version's last day in force was {ended} and the next took "
                        f"effect {starts}" if ended and starts else "")
                     + ". Call version_history for the full record.")}


def version_history(data_dir: str | Path, citation: str) -> list[dict]:
    norm = " ".join(citation.split())
    rows = _db(Path(data_dir)).execute(
        "SELECT * FROM docs WHERE citation=? ORDER BY effective_from", (norm,)).fetchall()
    return [{"path": r["path"], "effective_from": r["effective_from"],
             "effective_to": r["effective_to"], "is_current": bool(r["is_current"]),
             "sha": r["sha"]} for r in rows]


def list_programs(data_dir: str | Path) -> list[dict]:
    rows = _db(Path(data_dir)).execute(
        """SELECT program, COUNT(*) n FROM docs
           WHERE is_current=1 AND program != '' GROUP BY program ORDER BY n DESC"""
    ).fetchall()
    return [{"program": p, "current_sections": n} for p, n in rows]
