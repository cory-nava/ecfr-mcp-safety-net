"""Retrieval tiers for the ecfr-mcp MCP server.

Tier resolution (no flags, no config files):
  ECFR_MCP_TIER=free|best   -> pinned tier (override)
  ECFR_MCP_LLM_URL set      -> "best"  via a local OpenAI-compatible server (LM Studio)
  OPENROUTER_API_KEY set    -> "best"  (HyDE + LLM rerank over dense candidates)
  otherwise                 -> "free"  (citation fast-path + dense vectors)

Reranker model (best tier only):
  ECFR_MCP_RERANKER=luna        -> openai/gpt-6-luna  (default; 0.875 r@1)
  ECFR_MCP_RERANKER=flash-lite  -> google/gemini-2.5-flash-lite  (faster; 0.76 r@1)
  ECFR_MCP_RERANKER=<model-id>  -> any OpenRouter model ID

HyDE (Hypothetical Document Embeddings) candidate generation:
  Enabled by default; generates one CFR-style passage per query to improve
  candidate recall before reranking — especially effective for paraphrase queries.
  ECFR_MCP_HYDE=0               -> disable HyDE, use plain query vector
  ECFR_MCP_HYDE_MODEL=<id>      -> OpenRouter model for HyDE generation
                                    (default: google/gemini-2.5-flash-lite)

Local LLM (LM Studio, Ollama, llama.cpp: any OpenAI-compatible endpoint):
  ECFR_MCP_LLM_URL=<base-url>   -> e.g. http://localhost:1234/v1; HyDE and rerank both run
                                    there, OpenRouter is never called
  ECFR_MCP_LLM_MODEL=<id>       -> model for both (default: google/gemma-4-12b)
  ECFR_MCP_LLM_REASONING=<lvl>  -> reasoning_effort sent to the server (default: none;
                                    "off" sends nothing and lets the model decide)

"best" degrades gracefully: any LLM failure falls back to free-tier
ranking for that query rather than erroring.
"""
from __future__ import annotations

import functools
import json
import os
import re
import sqlite3
import time
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

CITATION_RE = re.compile(r"(\d+)\s+CFR\s+([\w.]+)", re.IGNORECASE)
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL_LUNA = "openai/gpt-6-luna"
RERANK_MODEL_FLASH = "google/gemini-2.5-flash-lite"
RERANK_API = "https://openrouter.ai/api/v1/chat/completions"
LOCAL_MODEL = "google/gemma-4-12b"
TOP_N = 50
BATCH = 10

_MODEL_ALIASES = {"luna": RERANK_MODEL_LUNA, "flash-lite": RERANK_MODEL_FLASH, "flash": RERANK_MODEL_FLASH}

HYDE_MODEL = "google/gemini-2.5-flash-lite"
HYDE_PROMPT = (
    "Write a short passage (4-6 sentences) in the style of a U.S. federal "
    "benefits regulation (Code of Federal Regulations) that answers the "
    "following question. Use formal regulatory terminology. Write only "
    "the passage, no disclaimers.\n\nQuestion: {q}"
)

RERANK_PROMPT = (
    "You are scoring passages for a legal researcher. Query: {q}\n\n"
    "Passages:\n{psgs}\n\n"
    "Rate each passage's relevance to the query from 0 (irrelevant) to 10 "
    "(directly answers it). Reply with ONLY a JSON object mapping passage "
    "number to score, e.g. {{\"1\": 8, \"2\": 3}}. No other text."
)


def _local_url(env=None) -> str:
    """Base URL of a local OpenAI-compatible server, or "" when unset."""
    e = env if env is not None else os.environ
    return (e.get("ECFR_MCP_LLM_URL") or "").strip().rstrip("/")


def _rerank_model(env=None) -> str:
    """Return the model ID to use for reranking."""
    e = env if env is not None else os.environ
    if _local_url(e):
        return (e.get("ECFR_MCP_LLM_MODEL") or LOCAL_MODEL).strip()
    raw = (e.get("ECFR_MCP_RERANKER") or "").strip()
    if raw:
        return _MODEL_ALIASES.get(raw.lower(), raw)
    return RERANK_MODEL_LUNA


def resolve_tier(env=None) -> tuple[str, str]:
    """Return (tier, reason). tier in {"free", "best"}."""
    e = env if env is not None else os.environ
    override = (e.get("ECFR_MCP_TIER") or "").strip().lower()
    if override in ("free", "best"):
        return override, "ECFR_MCP_TIER override"
    hyde = "disabled" if (e.get("ECFR_MCP_HYDE") or "1") == "0" else "enabled"
    if _local_url(e):
        return "best", f"ECFR_MCP_LLM_URL={_local_url(e)}, model={_rerank_model(e)}, hyde={hyde}"
    if e.get("OPENROUTER_API_KEY"):
        model = _rerank_model(e)
        return "best", f"OPENROUTER_API_KEY is set, reranker={model}, hyde={hyde}"
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


def _chat(prompt: str, model: str, api_key: str | None, temperature: float,
          max_tokens: int, timeout: float) -> tuple[str, dict]:
    """One chat completion against the local server or OpenRouter. Returns (text, usage).

    Falls back to the reasoning field when content is empty (reasoning models put
    their answer there when they run out of tokens).
    """
    local = _local_url()
    body = {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens}
    if local:
        url = f"{local}/chat/completions"
        headers = {"Content-Type": "application/json"}
        effort = (os.environ.get("ECFR_MCP_LLM_REASONING") or "none").strip().lower()
        if effort != "off":
            body["reasoning_effort"] = effort
        if effort != "none":
            # thinking models spend most of the budget before answering
            body["max_tokens"] = max(max_tokens, 4096)
        timeout = max(timeout, 300)  # local servers queue parallel requests
    else:
        url = RERANK_API
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {api_key}",
                   "HTTP-Referer": "https://github.com/ctrimm/ecfr-mcp",
                   "X-Title": "ecfr-mcp server"}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read().decode())
    msg = payload["choices"][0]["message"]
    return (msg.get("content") or msg.get("reasoning") or ""), payload.get("usage", {})


def _llm_scores(pairs: list[tuple[str, str]], api_key: str,
                call_api=None, model: str | None = None) -> list[dict]:
    """Score batches of (query, passage) listwise in parallel.

    Returns list of {num: score} dicts, one per batch, in order.
    Batches run concurrently via ThreadPoolExecutor (one at a time against a local
    server); each batch retries once on parse failure before propagating the exception.
    """
    if model is None:
        model = RERANK_MODEL_LUNA
    if call_api is None:
        def call_api(prompt: str) -> tuple[str, dict]:
            msg, usage = _chat(prompt, model, api_key, 0.0, 500, 180)
            mm = re.search(r"\{.*\}", msg, re.DOTALL)
            if not mm:
                raise RuntimeError("no JSON scores in model response")
            return mm.group(0), usage

    def _score_one(b_start: int) -> dict:
        chunk = pairs[b_start:b_start + BATCH]
        q = chunk[0][0]
        psgs = "\n".join(f"[{i + 1}] {p[:1500]}" for i, (_, p) in enumerate(chunk))
        prompt = RERANK_PROMPT.format(q=q, psgs=psgs)
        for attempt in range(2):
            try:
                raw, _ = call_api(prompt)
                return json.loads(raw)
            except Exception:
                if attempt == 0:
                    time.sleep(10)
                    continue
                raise

    starts = list(range(0, len(pairs), BATCH))
    if len(starts) == 1 or _local_url():
        # one request at a time on a local server: parallel batches swamp a laptop
        return [_score_one(s) for s in starts]
    out: list[dict | None] = [None] * len(starts)
    with ThreadPoolExecutor(max_workers=len(starts)) as ex:
        fmap = {ex.submit(_score_one, s): i for i, s in enumerate(starts)}
        for f in as_completed(fmap):
            out[fmap[f]] = f.result()
    return out  # type: ignore[return-value]


def _hyde_candidate_vec(corpus: Corpus, query: str, query_vec: np.ndarray,
                        api_key: str | None) -> np.ndarray:
    """Generate a hypothetical CFR passage and return query+passage averaged vector.

    Raises on any failure so the caller can fall back to query_vec transparently.
    """
    if _local_url():
        model = _rerank_model()
    else:
        model = (os.environ.get("ECFR_MCP_HYDE_MODEL") or HYDE_MODEL).strip()
    passage, _ = _chat(HYDE_PROMPT.format(q=query), model, api_key, 0.7, 400, 45)
    passage = passage.strip()
    if not passage:
        raise RuntimeError("empty HyDE passage")
    hp = corpus.embed(passage)
    avg = query_vec + hp
    return _normalized(avg.reshape(1, -1))[0]


def _llm_rerank(corpus: Corpus, query: str, candidate_vec: np.ndarray,
                key: str | None, top_k: int, model: str, call_api=None) -> list[Hit]:
    """Score top-N chunk candidates with the given model. Raises on any failure."""
    order = np.argsort(-(corpus.chunk_vecs @ candidate_vec))[:TOP_N]
    pairs = [(query, corpus.chunk_texts[i]) for i in order]
    batch_scores = _llm_scores(pairs, key, call_api=call_api, model=model)
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
    return [_hit_from_row(corpus.doc_row(p), best[p], r)
            for r, p in enumerate(ranked, 1) if corpus.doc_row(p)]


def search_best(corpus: Corpus, query: str, top_k: int = 5,
                api_key: str | None = None,
                call_api=None) -> tuple[list[Hit], bool]:
    """HyDE candidate generation + LLM rerank over top-50 chunks.

    Degradation chain: primary model → flash-lite fallback → free tier.
    Returns (hits, degraded): degraded=True when the primary model was not used.
    """
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    local = bool(_local_url())
    hit = _citation_hit(corpus, query)
    if hit:
        return [hit], False
    if not key and not local:
        return search_free(corpus, query, top_k), True
    # embedding errors are not LLM failures: let them surface instead of hiding behind the free tier
    q = corpus.embed(query)
    model = _rerank_model()
    # HyDE: average query vector with a hypothetical passage vector for better candidate recall.
    # Skipped when call_api is provided (test stub) or ECFR_MCP_HYDE=0.
    if call_api is None and os.environ.get("ECFR_MCP_HYDE", "1") != "0":
        try:
            candidate_vec = _hyde_candidate_vec(corpus, query, q, key)
        except Exception:
            candidate_vec = q
    else:
        candidate_vec = q
    try:
        return _llm_rerank(corpus, query, candidate_vec, key, top_k, model, call_api), False
    except Exception:
        pass
    # Primary model failed: fall back to flash-lite before dropping to free tier.
    # Only in real OpenRouter calls: test stubs skip this path, and a local setup never
    # sends queries to the cloud.
    if model != RERANK_MODEL_FLASH and call_api is None and not local:
        try:
            return _llm_rerank(corpus, query, candidate_vec, key, top_k, RERANK_MODEL_FLASH), True
        except Exception:
            pass
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
