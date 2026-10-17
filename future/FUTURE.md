# Future experiments

Ideas worth trying when there's a bigger box or more budget. Each entry:
what, why, what we know, and how to run it.

## EmbeddingGemma 2 bake-off

**What:** Rebuild the section vector index with `google/embeddinggemma-2`
(released 2026-10-06, 270M text-only / 740M multimodal, 768-dim, 8K context,
Apache 2.0) and measure dense retrieval on the frozen golden set against the
shipped `bge-small-en-v1.5` index.

**Why:** It's the newest open embedding model in this weight class. The
candidate pool is the bottleneck for the reranked tiers (round 8 showed
better candidates help exactly where the pool is weakest), so a stronger
embedder could lift r@3/r@5.

**What we know (2026-10-17):**
- Tried on the 7GB sandbox box: OOM-killed twice. The HF checkpoint is the
  full 740M multimodal model; even fp16 + batch 8 + 512-token cap dies after
  loading weights. Needs 16GB+ RAM.
- Press coverage notes v2's *text* scores "barely moved" vs v1; the big
  gains were on code retrieval. Temper expectations.
- Our bge-base (768-dim) vs bge-small experiment gave identical r@1
  (0.76); the gap only opened at r@3/r@5. The embedding is not the
  bottleneck — the LLM reranker is.

**How:** `bash future/embeddinggemma2_bakeoff.sh [corpus_dir] [out_dir]`
No API spend (dense eval only). For a rerank comparison afterwards, rebuild
chunk vectors with eg2 too (see `build/build_chunks.py`) and run
`golden/exp_rerank_candidates.py` with `--model-emb google/embeddinggemma-2`.

## Reranker v2-m3 (deferred, not disproven)

`bge-reranker-v2-m3` OOM-killed at 13/50 queries on the 7GB box
(`build/rerank_v2m3_results.jsonl` holds the partial 13). Retry on a bigger
box with memory fixes before judging it.

## Larger embedding models

At n=50, differences under ~0.04 are noise. Before hardening any embedding
win into lore, re-confirm on a larger query set.
