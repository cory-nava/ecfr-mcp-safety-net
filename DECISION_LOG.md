# Decision log

Data-driven decisions made while building ecfr-mcp. Each entry: the decision,
the evidence, the date. Decisions are revisited when new evidence contradicts them.

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

## 2026-10-12: Golden set frozen before retrieval existed

**Decision:** The 50-query golden eval set was authored and frozen
(2026-10-12) before any embedding ran or any retrieval result was seen.

**Evidence (methodological):** A test set tuned against the system it
measures is worthless. The set caught the prefix bug in round 1 — the
payoff of the discipline. Rule: add queries freely; never edit an existing
query to make the retriever pass.
