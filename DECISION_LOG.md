# Decision log

Data-driven decisions made while building ecfr-mcp. Each entry: the decision,
the evidence, the date. Decisions are revisited when new evidence contradicts them.

## 2026-10-23: Embedding bakeoff — bge-small stays; embedder is not the bottleneck

**Decision:** `BAAI/bge-small-en-v1.5` remains the shipped embedding model. No tested alternative
improved r@1 on the golden set; the bottleneck is the reranker, not the embedder.

**Evidence (golden set, n=50, dense + citation fast-path, Mac Studio M3 Ultra):**

| Model | r@1 | r@3 | r@5 | Encode time | Notes |
|---|---|---|---|---|---|
| bge-small-en-v1.5 (shipped) | 0.600 | 0.780 | 0.800 | 3.5 min (MPS) | baseline |
| bge-large-en-v1.5 | 0.600 | 0.780 | 0.860 | 32 min (CPU¹) | no r@1 gain |
| google/embeddinggemma-2 | — | — | — | impractical² | untested |
| nomic-embed-text-v1.5 | — | — | — | crashes³ | untested |
| Qwen3-Embedding-0.6B | — | — | — | ~12 h (MPS)⁴ | untested |

¹ bge-large ran before MPS support was added; with MPS it would be ~3 min, but r@1
  is still 0.600 — identical to bge-small.  
² EmbeddingGemma 2: crashes on MPS (MPSNDArrayMatrixMultiplication fails on
  4.9B-element per_layer_projection_norm tensor — 64-bit indexing not supported).
  CPU is infeasible because the model uses 8192-token context; quadratic attention
  over long CFR sections makes each batch take several minutes on M3 Ultra CPU.  
³ nomic-embed-text-v1.5: `AttributeError: NomicBertModel has no
  get_extended_attention_mask` with the current transformers version.  
⁴ Qwen3-Embedding-0.6B: decoder-only architecture, 584 s/batch on MPS → ~12 h total.
  Causal attention provides no inference speedup over CPU for embedding tasks.

**Why bge-small stays:**
- r@1 is the only metric that matters for the free tier (user sees the top result).
  bge-large ties at 0.600; no tested model beats it.
- The +6pp at r@5 from bge-large is real but irrelevant: the LLM reranker (Luna)
  reorders top-50 candidates anyway, so r@5 of the raw embedding pool doesn't
  drive end-to-end quality.
- bge-small: 127 MB download, 3.5 min encoding. bge-large: 335 MB, 32 min. A 2.6×
  heavier model for identical r@1 is not a trade-off worth making for users.
- The free-tier ceiling appears to be ~0.60–0.64 r@1 regardless of embedder. What
  moves the needle is the LLM reranker: Luna + HyDE brings that to 0.82 (golden) /
  0.81 (silver). Invest in the reranker, not the embedder.

Eval script: `future/embedding_bakeoff.sh` (single model or `--suite` mode).
Known omissions: nomic needs a transformers pin; EmbeddingGemma 2 needs CUDA;
Qwen3 needs a GPU with decoder-optimized inference (vLLM / TensorRT).

## 2026-10-23: Luna + HyDE made defaults; flash-lite fallback chain added; silver set created

**Decision:** Four changes shipped together to push default production quality to 0.81 r@1 on the
larger silver eval (was 0.76 with flash-lite + no HyDE):

1. **Luna (`gpt-6-luna`) is now the default reranker** (overturns the 2026-10-17 "flash-lite stays
   default" entry). Evidence: 0.875 vs 0.76 r@1 on golden; the cost delta (+25%, ~$0.002/query) is
   acceptable given the recall gap. Flash-lite remains available via `ECFR_MCP_RERANKER=flash-lite`.

2. **HyDE is now enabled by default** (overturns the 2026-10-17 "server default stays
   query-candidates" entry). Evidence: HyDE + Luna on silver = 0.81 vs 0.76 query-candidates +
   flash-lite; the one extra LLM call (~$0.0001, ~5s) is worth it. Disable with `ECFR_MCP_HYDE=0`.
   HyDE model configurable via `ECFR_MCP_HYDE_MODEL` (default: `google/gemini-2.5-flash-lite`).

3. **Flash-lite fallback chain added** (Luna → flash-lite → free tier on failure). Per-batch failure
   rate ~4.5% → P(≥1 of 5 parallel batches fails) ≈ 21%, consistent with 49–62 degraded queries
   per 236 in silver runs. Without the fallback, those queries collapsed to free-tier (0.51 r@1);
   with flash-lite as the fallback they land at ~0.76. Overall silver r@1 lifted from 0.76 → 0.81.

4. **Silver eval set generated** (`silver/eval.json`, 236 queries, 50% paraphrase, 15 programs),
   via `silver/generate.py` using OpenRouter + gemini-2.5-flash-lite. No exact-citation queries
   (the citation fast-path is already tested by the golden set). Skewed toward paraphrase to
   stress-test the hardest failure mode (golden paraphrase r@1 was 0.42 free tier).

**Evidence (silver set, n=236, 50% paraphrase, best tier, 2026-10-23):**
- Overall: r@1=0.81, r@3=0.88, r@5=0.90
- Natural (n=118): r@1=0.92, r@3=0.97, r@5=0.97
- Paraphrase (n=118): r@1=0.69, r@3=0.79, r@5=0.82 (+41pp over free tier on golden)
- 62 queries degraded (Luna → flash-lite); those land at ~0.76 r@1 instead of ~0.51

Golden set (n=50): r@1=0.82, r@3=0.98, r@5=0.98, degraded=0. n=50 is too small for reliable
paraphrase sub-scores (95% CI ≈ ±0.09); silver n=118 is the authoritative paraphrase number.

## 2026-10-22: get_section misses say which case they are; as_of is validated

**Decision:** A `get_section` miss returns one of five statuses instead of treating every
miss with history as "repealed": `unknown_citation`; `invalid_date`; `not_in_force` (no
version is current: repealed / [Reserved] / renumbered, the 2026-10-20 behaviour, fields
unchanged); `not_in_force_on_date` (the date falls in a gap, with `gap_from`/`gap_to`);
`before_recorded_history` (earlier than the section's first recorded version). `as_of`
must be a real `YYYY-MM-DD` date.

**Evidence:** On 604b0a4, `get_section("7 CFR 273.9", as_of="1990-01-01")` (SNAP income,
in force) said "No current text… likely repealed", and so did 42 CFR 457.960 on
2025-01-01 (removed 2024-06-03, restored 2026-06-03). Worse, dates compare as strings and
were never validated: `as_of="2024-6-1"` returned the November 2024 version (wrong; the
2023-09-26 version was in force) and `as_of="June 2024"` returned today's text, both with
`found: true`. History in the corpus starts at the version in force on 2024-01-01, so an
earlier date is a gap in our records, not evidence about the CFR. Tests cover each case
and fail on 604b0a4. Note: `as_of` is this project's own parameter; OKF (v0.1 or v0.2)
has no point-in-time concept, and `effective_from`/`effective_to` are extension keys.

## 2026-10-20: Round 8 filed; external retrieval findings kept separate

**Measured here (Round 8, 2026-10-17, frozen 50-query golden set):**
HyDE-averaged top-50 chunk candidates + listwise LLM rerank is the best
measured config at 0.76/0.92/0.94, $0.088/50 queries (full entry below,
dated 2026-10-17). Hybrid dense∪BM25 candidates hurt (0.70/0.84/0.88);
dense top-100 was a wash at 2x cost (0.76/0.86/0.92).

**External findings (web discovery report, 2026-10-18 — NOT measured on
this corpus; recorded as calibration for the next embedder/reranker
choice, not as results):**

- On fact-lookup queries, HyDE wins cleanly (R@10 0.79 vs baseline 0.71).
- Query decomposition wins on multi-hop queries.
- Naive HyDE-plus-encoder hybrids backfire (-10.8pp NDCG@5).
- The same report puts `bge-reranker-v2-m3` at Cohere Rerank quality on
  CPU at zero API cost. Untested here: its only run on this corpus
  OOM-killed at 13/50 queries on a 7GB box (partial scores in
  `build/rerank_v2m3_results.jsonl`). It stays a candidate, not a result.

## 2026-10-17: server default stays query-candidates (zero extra calls)

**Decision:** The best tier keeps query-candidate + rerank as the default.
HyDE-averaged candidates (0.76/0.92/0.94, the best measured config) stay
opt-in: one extra flash call per query (~$0.0001, 5–10s) for +4pp at
r@3/r@5 with r@1 unchanged is a latency/cost tradeoff for the deployer,
not the default. The README leaderboard records all five measured
configs with per-query costs.

## 2026-10-17: HyDE-averaged candidates + rerank is the new best (0.76/0.92/0.94)

**Decision:** The best measured config is now HyDE-averaged top-50 chunk
candidates + listwise LLM rerank: 0.76/0.92/0.94 at $0.088/50 queries —
same cost as the query-candidate baseline. Whether it becomes the server
default is open: it adds one HyDE flash call per query (~$0.0001, 5–10s
latency) for +4pp at r@3/r@5 with r@1 unchanged.

**Evidence:** Round 8 (`golden/exp_rerank_candidates.py`, same frozen 50).
Hybrid dense∪BM25 candidates HURT (0.70/0.84/0.88; paraphrase r@1
0.583→0.417 — lexical distractors fool the LLM scorer, same mechanism as
the bge-reranker-base failure). Dense top-100 is a wash (0.76/0.86/0.92,
2x cost). This refines round 7's "combo does not stack": on bge-base the
combo lost everywhere, but on the weaker bge-small pool HyDE-averaged
candidates gain where the pool is thinnest. Total experiment spend this
session: ~$0.49.

## 2026-10-17: best tier re-measured on bge-small (0.76/0.88/0.90)

**Decision:** Publish best-tier numbers measured on the shipped bge-small
candidates; retire the "old bge-base figure" label.

**Evidence:** Same frozen 50, `golden/exp_llm_rerank.py`, gemini-2.5-flash-lite,
$0.086 for the run (scores in `golden/llm_rerank_scores_bgesmall.jsonl`).
r@1 identical to bge-base (0.76); the gap opens at r@3/r@5 (0.88/0.90 vs
0.94/0.96) — bge-small's top-50 misses the gold section outright more often.
Same lesson as the free tier: the candidate generator, not the reranker, is
where the ~4pp went.

## 2026-10-17: bge-small everywhere, indexes rebuilt, open-source cleanup

**Decision:** `BAAI/bge-small-en-v1.5` (384-dim) is the model for both the
section and the chunk index. `data/manifest.json` records the model and the
server loads whatever it names; the server refuses to start if the indexes
disagree on dimensions.

**Evidence:** The 2026-10-13 switch to bge-small rebuilt `vectors.npz` but
not `chunks.npz` (still 768-dim bge-base). Every best-tier query hit a shape
mismatch, which the degrade-to-free handler swallowed, so the best tier had
silently been the free tier since then. The same rebuild appended to the old
`meta.sqlite` instead of replacing it (every row twice, `is_current` stored as
text so `bool('0')` was true). All rebuilt from the refreshed corpus; free
tier re-measured with the shipped code (`golden/eval_server.py`): 0.60 / 0.78 /
0.80, vs 0.64 / 0.78 / 0.84 for bge-base on the same refreshed corpus
(2 of 50 queries at r@1 and at r@5; within noise at n=50, but a real cost to
weigh against the 70% smaller model). Also: `as_of`
lookups now treat `effective_to` as the last day in force (inclusive), as
ecfr-markdown writes it; the corpus tarball is read into memory rather than
extracted next to the code (no lock, works on Windows).

## 2026-10-12: Production retrieval = citation fast-path + vectors (BM25 not fused)

**Decision:** The ranking path is deterministic citation lookup (queries
matching `\d+ CFR [\w.]+` resolve directly against current docs) plus
bge-base-en-v1.5 cosine ranking. The BM25 index is built and kept in
`meta.sqlite` as a fallback/debug leg, but RRF fusion is NOT in the
ranking path.

**Evidence (reverses the earlier BM25-first call):** Golden eval round 3,
same frozen 50 queries. BM25 alone: 0.28/0.44/0.54. RRF(BM25, vectors)
k=5..60: best 0.48/0.76/0.84 — never beats vectors + citation fast-path at
0.64/0.78/0.84 on any k. BM25 adds noise on natural/paraphrase queries
(RRF r@1 0.48 vs vectors+cite 0.64). The one thing BM25 was for — exact
citations — the fast-path does perfectly (1.00) by construction, and
correctly: a citation query is a lookup, not retrieval. Weighted fusion was
deliberately not tried: tuning weights on n=50 would be overfitting; it
needs a larger query set first. Caveat: n=50 is small; revisit if the query
distribution shifts. Remaining headroom is paraphrase r@1 (0.417).

## 2026-10-12: Index current versions only

**Decision:** `--as-of-` version files are recorded in `meta.sqlite`
(`is_current=0`, direct lookup still works) but excluded from the vector
index.

**Evidence:** Near-duplicate versions of the same section would crowd the
top-k with redundant hits. 1,028 of 3,306 corpus files are version files;
indexing them nearly doubles the index for no retrieval gain. Temporal
questions are served by direct version lookup, not search.

## 2026-10-12: Front matter excluded from embed text

**Decision:** Embed `title + body` only; YAML front matter is never embedded.

**Evidence:** Front matter contains retrieval-gaming keys (`tags: [snap,
cfr, regulation]`, citation strings). Embedding them would let queries match
on metadata rather than content, inflating scores without improving answer
quality.

## 2026-10-12: Local bge-base-en-v1.5 (768-dim), CPU

**Decision:** Embeddings from `BAAI/bge-base-en-v1.5`, run locally on CPU,
instead of API embeddings.

**Evidence:** Corpus is ~5M tokens: ~$0.10 via API vs. $0 local. Local keeps
the build reproducible with no spend approval and no network dependency at
query time. Measured build time on this machine: ~57 min for 2,278 docs
(one-time cost).

## 2026-10-12: No instruction prefixes (query:/passage:)

**Decision:** Embed plain `title + body` for docs and raw query text for
queries — no `query:`/`passage:` prefixes.

**Evidence:** Golden eval round 1 (prefixes on): exact-citation queries
ranked ~1270th of 2,278 (worse than random); overall recall@1 0.40.
`bge-base-en-v1.5` was not trained with instruction prefixes, so both sides
landed out-of-distribution. Spot-check without prefixes: exact query
1270 -> 316, natural query 11 -> 6. Full round-2 eval pending on the
prefix-free rebuild.

## 2026-10-12: Brute-force cosine, no ANN index

**Decision:** Retrieval is exact brute-force cosine over L2-normalized
vectors (numpy), no HNSW/ANN library.

**Evidence:** 2,278 x 768-dim vectors = 6.5MB; a full scan is microseconds.
An ANN index would add a dependency and approximate results for zero
measurable latency gain at this scale.

## 2026-10-12: Exact citations are a keyword job

**Decision:** The hybrid design routes exact-citation queries to BM25;
vectors are not expected to solve them.

**Evidence:** Even without the prefix bug, the query "7 CFR 273.9" ranked
its section 316th by dense similarity — a 4-token citation carries almost
no semantic signal. BM25 matches it trivially. The golden set's 5 `exact`
queries exist to keep this honest.

## 2026-10-12: Corpus decoupled from the server repo

**Decision:** The corpus (~21MB, 3,384 files) is not vendored into the
server repo. Setup seeds a local cache from a versioned, checksummed
release zip — one download, then fully offline.

**Evidence:** Repo code is 33KB; forcing a 3,384-file git clone as an MCP
install step is a packaging failure. One-time 21MB download (often much
less zipped) is a non-issue; per-query GitHub dependence would be.

## 2026-10-13: MCP server built with env-var tiers (Cory's design)

**Decision:** Ship the server now with tier resolution purely from the
environment: `ECFR_MCP_TIER` override > `OPENROUTER_API_KEY` present >
free. No flags, no config files. Best tier degrades to free-tier ranking
on any LLM failure rather than erroring. Corpus data ships in `data/`
(64MB) instead of the deliberated release-zip seed — simpler, offline
immediately; revisit if repo weight hurts.

**Evidence:** Free tier smoke-tested over real MCP stdio (all 4 tools:
citation lookup, natural search, full-text section incl. as_of versioning,
version history, program list). Best tier tested with stubbed LLM:
scoring path, degradation fallback, citation bypass of the LLM.
Corpus ships as `data/corpus.tar.gz` (3.4MB) instead of 3,399 loose
files — a single archive is also far kinder to git hosting than
thousands of small blobs (verified single-blob uploads are fine:
2MB/2.7s, 17MB/12s).
Server extracts the tarball once, on first `get_section`, under a lock.

## 2026-10-13: HyDE+rerank combo rejected (does not stack)

**Decision:** Query-vector candidates + LLM rerank stays the best config.
HyDE-averaged candidates + rerank scored 0.74/0.90/0.94 vs 0.76/0.94/0.96
— worse everywhere, plus an extra LLM call per query for nothing.

**Evidence:** Same frozen 50, $0.09 for the combo run. The reranker, not
the candidate generator, is doing the work.

## 2026-10-13: LLM-judge rerank is the new best (0.76/0.94/0.96)

**Decision:** Listwise LLM rerank (gemini-2.5-flash-lite, top-50 chunks)
beats HyDE+query: +4pp r@1, +12pp r@3, natural r@3/r@5 perfect. Becomes
the measured-best config. Same adoption caveat as HyDE (per-query LLM
cost/latency/dependency — now ~6 calls/query, ~$0.002) deferred to server
build.

**Evidence:** Same frozen 50. $0.09 total. Scores in
`golden/llm_rerank_scores.jsonl`; eval reproducible locally with no API.
Model note: deepseek-v4-flash via SiliconFlow returned intermittent
null-content 200s in streaks and was abandoned after 5/5 clean on gemini.

## 2026-10-13: HyDE wins the retrieval round; adoption deferred to server build

**Decision:** HyDE passage + query vector averaging (0.72/0.82/0.88) is the
measured-best retrieval config, beating whole-section vectors (0.64/0.78/
0.84) by +8pp r@1. It is NOT yet the production default — that call waits
for the MCP server build, where per-query LLM latency (~5-10s), cost
(~$0.0001), and the OpenRouter dependency get weighed against the recall
gain. The losing configs (rerank-base, chunking, BM25) stay out.

**Evidence:** Same frozen 50 queries. HyDE-only also beats base
(0.68/0.84/0.90); the average is best-balanced (natural 0.758, paraphrase
0.500). Total experiment cost: $0.0056.

## 2026-10-12: Chunking measured as neutral; not adopted into production path

**Decision:** Keep whole-section vectors as the production index. The
chunked index (6,222 chunks, build/build_chunks.py) is built and retained
for future experiments but not served.

**Evidence:** Same frozen 50 queries: chunked best-chunk + citation =
0.62/0.80/0.86 vs whole-section 0.64/0.78/0.84. Within noise; helps natural
queries slightly (+0.03 r@1), hurts paraphrase r@1 (0.417 -> 0.250).
Truncation is real (39.7% of docs > 512 tokens) but best-chunk aggregation
doesn't convert the fix into recall gains on this query set.

## 2026-10-12: bge-reranker-base rejected (both configurations)

**Decision:** No cross-encoder rerank stage with bge-reranker-base.

**Evidence:** Rerank top-50 sections: 0.52/0.70/0.74. Rerank top-50 chunks:
0.52/0.70/0.76. Both well below the 0.64/0.78/0.84 baseline. The obvious
confound (reranker truncated to each candidate's head) was eliminated by
the chunk variant — identical degradation, so the cause is model/domain
mismatch, not truncation. Not retried with v2-m3 yet (heavier; named
follow-up, not assumed win).

## 2026-10-12: Golden set frozen before retrieval existed

**Decision:** The 50-query golden eval set was authored and frozen
(2026-10-12) before any embedding ran or any retrieval result was seen.

**Evidence (methodological):** A test set tuned against the system it
measures is worthless. The set caught the prefix bug in round 1 — the
payoff of the discipline. Rule: add queries freely; never edit an existing
query to make the retriever pass.

## 2026-10-17 — Luna (gpt-6-luna) is the best reranker by a wide margin; Haiku-5.5 rejected

Compared three LLM rerankers on HyDE-averaged candidates (bge-small), same 50
queries: gemini-2.5-flash-lite 0.76/0.92/0.94, **gpt-6-luna 0.875/1.000/1.000**,
claude-haiku-5.5 unusable (0/50 scored — empty/null responses on 15KB prompts).
On the 48 queries Luna completed, flash-lite scores 0.729/0.917/0.938, so the
gap is not a dropout artifact. Luna emits ~80 reasoning tokens/call; the
chain-of-thought is the quality edge and the latency cost (30–40s/batch vs
1–2s). Cost $0.0022/query vs $0.0018 (+25%). Results in
`golden/results-2026-10-17.md`, scores in `golden/llm_rerank_scores_luna.jsonl`.
Server default stays flash-lite (speed); Luna documented as the quality option.

## 2026-10-17 — gpt-5.4-nano rejected as reranker (0.58/0.88/0.94)

Tested openai/gpt-5.4-nano (latest nano-tier, $0.20/$1.25 per MTok) as listwise
reranker on HyDE-averaged candidates: r@1 0.58 vs flash-lite's 0.76 (-18pp),
paraphrase r@1 0.333. Fast (~2s/batch, 50/50 scored, $0.170) but a clearly
worse judge. Newer/pricier != better at relevance scoring. Ranking:
Luna (0.875) > flash-lite (0.76) > 5.4-nano (0.58) > Haiku-5.5 (unusable).
Flash-lite stays the server default.

## 2026-10-20 — get_section not_in_force re-applied to ecfr_mcp/ layout; da52c02 is_current claim corrected

**Decision:** Re-applied the `not_in_force` improvement from da52c02 (2026-10-13, old
`mcp-server/` layout) to the current `ecfr_mcp/` layout. `get_section` on a citation with
historical versions but no current text now returns `found: false, status: "not_in_force"`
with `versions`, `last_effective_from/to`, and a note pointing at `version_history` —
instead of a bare not-found indistinguishable from a typo. MCP tool description updated
to document the behavior. No golden re-measure: the search path is untouched and the
frozen golden set contains zero queries against reserved sections.

**Correction to da52c02:** its second claim — "`is_current` was `bool()` on a TEXT column
('0'/'1'), so it was always True even for superseded versions" — does NOT reproduce
against the current tree. `data/meta.sqlite` declares `is_current INTEGER`
(`build/build_vectors.py` writes Python ints 0/1; 1,029 rows 0 / 2,278 rows 1), so
`bool()` is already correct, and AGENTS.md codifies "`is_current` is an integer;
report it with `bool()`". That half of da52c02 was deliberately NOT ported — porting
`str(...) == "1"` would have deviated from the house rule for zero behavior change.

**Why the re-apply was needed:** da52c02 lives only on the stale local `master` (old
layout). The PR #1 restructure (`mcp-server/` -> `ecfr_mcp/`, commit 4b2fae6) dropped the
change — the current tree had no `not_in_force` handling. Pushing da52c02 as-is would
have pushed the old layout against the new tree.

**Evidence:** Direct tests against `data/meta.sqlite`: `45 CFR 260.10` -> not_in_force
(2 versions, last 2026-07-31 to 2026-09-28); `45 CFR 96.80` -> not_in_force; `as_of=
2026-08-15` on 260.10 returns the historical text with `is_current: false`; current
section `45 CFR 260.20` returns text with `is_current: true`; unknown citation
`45 CFR 999.999` still bare `found: false`. `python -m unittest discover -s tests -t .`:
12 passed, 1 error — the error is `tests/test_smoke.py` failing to `import mcp`
in an environment without the package installed; full stdio smoke test unverified
in that environment, same caveat as 2026-10-13. (Resolved 2026-10-22: the full
suite, smoke test included, passes 13/13 in a complete environment.)
