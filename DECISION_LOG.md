# Decision log

Data-driven decisions made while building ecfr-mcp. Each entry: the decision,
the evidence, the date. Decisions are revisited when new evidence contradicts them.

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
files — the 10-worker blob uploader hung through the sandbox egress
proxy; verified single-blob uploads are fine (2MB/2.7s, 17MB/12s).
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
