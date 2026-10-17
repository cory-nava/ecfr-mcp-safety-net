# Retrieval over the eCFR: Seven Rounds of Measured Experiments

**Technical whitepaper — ecfr-mcp retrieval bake-off, 2026-10-12/13;**
**2026-10-17 addendum: embedding model comparison (bge-small vs bge-base)**

## Abstract

We built a retrieval layer over a Markdown corpus of the U.S. Code of Federal
Regulations (2,278 current-version benefit-program sections) and measured
nine configurations against a single frozen 50-query golden set. The
simplest production baseline (deterministic citation lookup + dense vectors)
beat BM25, hybrid fusion, chunking, and a cross-encoder reranker. Two
LLM-assisted techniques then moved the needle: HyDE query expansion
(+8pp recall@1) and listwise LLM reranking (+12pp recall@1 over baseline,
natural-query recall@3/5 perfect). Total experiment spend: about $0.18.
Every number below is measured on the same frozen set — no query was ever
edited to make a system pass.

**2026-10-17:** the shipped index now uses `bge-small-en-v1.5` (384-dim,
127 MB) instead of `bge-base-en-v1.5` (768-dim, 418 MB). The free tier was
re-measured for both models on a refreshed corpus snapshot: bge-small
0.60/0.78/0.80 vs bge-base 0.64/0.78/0.84 — a ~4pp r@1 cost for half the
RAM. The paid tier was re-measured on bge-small the same day: 0.76/0.88/
0.90 ($0.086 for the 50-query run) — r@1 identical to bge-base, the gap
opening at r@3/r@5 where the weaker candidate pool misses outright.

## 1. Corpus and method

- **Corpus:** `ctrimm/ecfr-markdown` — the eCFR as Markdown, one file per
  section. Rounds 1–7 ran on the 2026-10-08 snapshot; the 2026-10-17
  model comparison on the 2026-10-15 snapshot. Indexed subset: 2,278
  current-version sections across benefit programs (SNAP, WIC, TANF,
  Medicaid/CHIP, LIHEAP, SSI, housing vouchers, Lifeline, child care, UI,
  school meals, Summer EBT).
- **Embeddings:** shipped index is `BAAI/bge-small-en-v1.5` (384-dim),
  brute-force cosine, no instruction prefixes. Round 1 (on bge-base)
  showed bge v1.5 was not trained with `query:`/`passage:` prefixes and
  they actively hurt (exact-citation queries ranked ~1270th of 2,278,
  worse than random); the no-prefix finding carries over.
- **Golden set:** 50 queries frozen *before any retrieval ran* — 5 exact
  citations ("7 CFR 273.9"), 33 natural questions, 12 paraphrases,
  stratified across 14 program areas. Sealed: queries are added, never
  edited to fit the system.
- **Metric:** recall@1 / @3 / @5, overall and per query type.

## 2. The production baseline

A query matching `\d+ CFR [\w.]+` is a **deterministic lookup**, not
retrieval — a query that IS a citation resolves by key (recall 1.00 by
construction, and correct by design). Everything else ranks by cosine
similarity over whole-section vectors. BM25 (SQLite FTS5, title 3x
weight) was built and kept as a fallback/debug leg, but RRF fusion never
beat vectors+citation on any k, so BM25 is not in the ranking path.

## 3. Seven rounds (chart: progression.png, recall_all.png)

All rounds below ran on bge-base (768-dim), corpus snapshot 2026-10-08:

| Round | System | r@1 | r@3 | r@5 | Cost |
|---|---|---|---|---|---|
| 2 | Vectors, no prefixes | 0.540 | 0.720 | 0.800 | $0 |
| 3a | BM25 alone | 0.280 | 0.440 | 0.540 | $0 |
| 3b | RRF hybrid (best k) | 0.500 | 0.760 | 0.840 | $0 |
| 3c | **Vectors + citation fast-path** | **0.640** | **0.780** | **0.840** | $0 |
| 4a | Chunked index (6,222 chunks, best-chunk) | 0.620 | 0.800 | 0.860 | $0 |
| 4b | bge-reranker-base, sections | 0.520 | 0.700 | 0.740 | $0 |
| 4c | bge-reranker-base, chunks | 0.520 | 0.700 | 0.760 | $0 |
| 5a | HyDE passage only | 0.680 | 0.840 | 0.900 | $0.006 |
| 5b | **HyDE + query averaged** | **0.720** | **0.820** | **0.880** | $0.006 |
| 6 | **LLM rerank, query candidates** | **0.760** | **0.940** | **0.960** | $0.09 |
| 7 | HyDE candidates + LLM rerank | 0.740 | 0.900 | 0.940 | $0.18 |

Per-type recall@1 for the three leaders (chart: recall_by_type.png):

| System | natural (n=33) | paraphrase (n=12) | exact (n=5) |
|---|---|---|---|
| Vectors + citation | 0.667 | 0.417 | 1.000 |
| HyDE + query | 0.758 | 0.500 | 1.000 |
| LLM rerank | 0.788 | 0.583 | 1.000 |

LLM rerank additionally reaches natural r@3/r@5 = **1.000** (perfect).

### 3b. 2026-10-17: bge-small vs bge-base (free tier)

Same frozen 50 queries, same code, refreshed corpus (2026-10-15); only
the embedding model differs. Full detail in
`golden/results-2026-10-17.md`.

| model | r@1 | r@3 | r@5 |
|---|---|---|---|
| bge-base (768-dim, rounds 1–7) | **0.64** | **0.78** | **0.84** |
| bge-small (384-dim, shipped) | 0.60 | **0.78** | 0.80 |

By query type, r@1 / r@3 / r@5:

| model | exact (n=5) | natural (n=33) | paraphrase (n=12) |
|---|---|---|---|
| bge-base | 1.00 / 1.00 / 1.00 | 0.667 / 0.788 / 0.848 | 0.417 / 0.667 / 0.750 |
| bge-small | 1.00 / 1.00 / 1.00 | 0.606 / 0.788 / 0.818 | 0.417 / 0.667 / 0.667 |

bge-base reproduces its 2026-10-12 figures exactly on the refreshed
corpus, so the eval is stable across the snapshot refresh and the gap
above is the model. The switch costs 2 queries at r@1 (natural
0.667 → 0.606) and 2 at r@5 — within noise at n=50, but pointing the
same way in every slice where the two differ. bge-small was chosen for
size: 127 MB vs 418 MB download, about half the RAM. Whether ~4pp of
r@1 is worth it is a product call; reverting is a rebuild of `data/`
with `--model BAAI/bge-base-en-v1.5`.

The paid tier (LLM rerank over chunk candidates) was re-measured on
bge-small 2026-10-17: 0.76/0.88/0.90 ($0.086; scores in
`golden/llm_rerank_scores_bgesmall.jsonl`). r@1 matches bge-base exactly;
the ~6pp gap at r@3/r@5 is the candidate pool, not the reranker.

## 4. Findings

1. **Prefixes hurt.** bge-base-en-v1.5 was not trained with `query:` /
   `passage:` prefixes; both sides landed out-of-distribution. Removing
   them: 0.40 -> 0.54 r@1.
2. **A citation is a lookup.** Short citation strings carry almost no
   semantic signal; dense retrieval ranked "20 CFR 416.1100" 592nd. The
   deterministic fast-path fixes the whole category.
3. **BM25-first loses.** RRF fusion (k=5..60) never beats vectors +
   citation on any cutoff. BM25 stays as fallback only.
4. **39.7% of sections exceed bge's 512-token limit** and were silently
   truncated — real data loss, but chunking (450-token windows, 64
   overlap, title-prepended, section = best chunk) measured as a wash
   (0.62/0.80/0.86): it fixed truncation in principle while diluting
   topical signal on paraphrase queries (r@1 0.417 -> 0.250).
5. **bge-reranker-base hurts in both configurations** (0.52/0.70/0.74 on
   sections, 0.52/0.70/0.76 on chunks). Reranking chunks eliminated the
   truncation confound, so the cause is model/domain mismatch — MS MARCO
   training rewards surface lexical overlap and promotes distractor docs
   (e.g. the OMB control-number list that cites everything).
6. **HyDE is the first real win** (+8pp r@1, $0.0056 total): a flash model
   writes a 4-6 sentence CFR-style passage answering the query; embedding
   the passage (alone, or averaged with the query vector) beats the query
   vector alone. HyDE-only is strongest on paraphrase (0.750); the
   average is the balanced best.
7. **Listwise LLM reranking wins overall** (gemini-2.5-flash-lite scores
   top-50 chunks 10-at-a-time, section = best chunk): 0.76/0.88/0.90 on
   bge-small for $0.086 (2026-10-17; 0.76/0.94/0.96 on bge-base,
   2026-10-13). Operational note: deepseek-v4-flash via SiliconFlow
   returned intermittent null-content HTTP 200s in streaks and was
   abandoned after gemini went 5/5 clean on identical prompts.
8. **The combo does not stack.** HyDE candidates + LLM rerank scored
   0.74/0.90/0.94 — worse everywhere than query candidates + rerank. The
   reranker does the heavy lifting; once it sees the top-50, the
   candidate source barely matters. Rejected (extra LLM call, no gain).
9. **Smaller embeddings cost ~4pp r@1** (2026-10-17). bge-small-en-v1.5
   (384-dim) trails bge-base 0.60 vs 0.64 r@1 on the same frozen set,
   identical at r@3. Shipped anyway for the size win (127 MB vs 418 MB,
   ~half the RAM) — a product tradeoff, revertible with one rebuild.

## 5. Cost analysis (chart: cost_frontier.png)

The efficiency frontier is steep and short: $0.18 of total experiment
spend bought +12pp r@1 / +16pp r@3 over the free baseline. Per-query
serving costs at the winning config: ~6 flash-model calls, roughly
$0.002 and 30-60s sequential latency (batchable). Whether that belongs
in the default serving path is a product decision — see deployment
tiers below.

## 6. Deployment: env-var-gated tiers (MCP server)

The server starts in the tier its environment supports — no flags, no
config files:

- **Free tier** (no key): citation fast-path + local vectors
  (bge-small as shipped). 0.60/0.78/0.80, fully offline after install,
  $0/query.
- **Best tier** (`OPENROUTER_API_KEY` set in the user's shell): adds the
  listwise LLM rerank over top-50 chunks. 0.76/0.88/0.90 on the shipped
  bge-small index (2026-10-17). ~$0.002/query.
- **Override:** `ECFR_MCP_TIER=free|best` pins a tier regardless of the
  key (e.g. force free for latency-sensitive use, or test best without
  exporting a key globally).

Degradation is graceful: if the key is absent or the LLM call fails, the
server serves the free tier rather than erroring. The golden set and all
experiment scripts ship in the repo so any tier choice stays
re-measurable.

## 7. Limitations

n=50 is small — differences under ~0.04 are noise, and several "wins"
should be re-confirmed on a larger set before hardening into lore. All
embedding work ran CPU-only on a 7GB box (bge-reranker-v2-m3 OOM-killed
there; the LLM judge replaced it). Two corpus snapshots (eCFR
2026-10-08 for rounds 1–7, 2026-10-15 for the model comparison);
version-chaining retrieval is built but not evaluated here.

## 8. Reproducibility

Repo: `ctrimm/ecfr-mcp` (private). `golden/eval.json` (frozen set),
`golden/eval.py`, `golden/eval_chunks.py`, `golden/exp_rerank.py`,
`golden/exp_rerank_chunks.py`, `golden/exp_hyde.py`,
`golden/exp_llm_rerank.py`, `golden/eval_server.py` (free-tier numbers
via the server's own code); `golden/results-2026-10-12.md` (rounds 1–7,
bge-base), `golden/results-2026-10-17.md` (bge-small vs bge-base);
`DECISION_LOG.md` (each data-backed decision with evidence and date).
