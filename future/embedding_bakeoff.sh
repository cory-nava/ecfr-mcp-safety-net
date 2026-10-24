#!/usr/bin/env bash
# Embedding-model bake-off for ecfr-mcp.
#
# Builds section vectors with any sentence-transformers-compatible model and
# evaluates dense retrieval on the frozen golden set (free-tier equivalent:
# citation fast-path + cosine).  No API spend.
#
# Usage — single model:
#   bash future/embedding_bakeoff.sh <hf-model-id> [corpus_dir] [out_dir]
#
# Usage — full recommended suite:
#   bash future/embedding_bakeoff.sh --suite [corpus_dir] [out_dir]
#
# Recommended suite (SUITE array below):
#   BAAI/bge-small-en-v1.5          127 MB  shipped baseline
#   BAAI/bge-large-en-v1.5          335 MB  same family, larger
#   nomic-ai/nomic-embed-text-v1.5  137 MB  competitive at bge-small size
#   Qwen/Qwen3-Embedding-0.6B       ~1.2 GB strong MTEB, heavier download
#   google/embeddinggemma-2         ~1.5 GB Google DeepMind, 768d, MTEB 68.46
#
# REQUIREMENTS: RAM scales with model size.  bge-large fits in ~4 GB fp16;
# Qwen3-Embedding-0.6B and embeddinggemma-2 need ~8 GB.
# No API spend: dense eval only.
#
# Output per model:
#   <out_dir>/vectors_<slug>.npz   section vectors (vectors, paths)
#   stdout: recall@1/3/5 overall and per query type
# Suite mode prints a comparison table at the end.

set -euo pipefail

# ── Recommended suite ────────────────────────────────────────────────────────
SUITE=(
  "BAAI/bge-small-en-v1.5"           # shipped baseline
  "BAAI/bge-large-en-v1.5"           # direct upgrade, same family
  "google/embeddinggemma-2"          # Google DeepMind, 768d, MTEB 68.46
  # nomic-ai/nomic-embed-text-v1.5 removed: fails with current transformers
  #   (AttributeError: NomicBertModel has no get_extended_attention_mask)
  # Qwen/Qwen3-Embedding-0.6B removed: decoder-only arch, ~12h/run on MPS
  #   (584s/batch × 73 batches — causal attention gives no inference speedup)
)

# Per-model prompt names for asymmetric embedding models.
# Using case statements instead of declare -A for bash 3.x compatibility (macOS default).
_doc_prompt()   { case "$1" in google/embeddinggemma-2) echo "Document"    ;; *) echo "" ;; esac; }
_query_prompt() { case "$1" in google/embeddinggemma-2) echo "SearchQuery" ;; *) echo "" ;; esac; }
# MPS batch size override: embeddinggemma-2 has a 12288 hidden dim and hits
# MPSNDArrayMatrixMultiplication "too large for kernel" at batch_size=32.
_mps_batch()    { case "$1" in google/embeddinggemma-2) echo "4"           ;; *) echo "32" ;; esac; }
# Models that must use CPU: MPS hits 64-bit indexing limits on large internal
# tensors (numel > ~2B). embeddinggemma-2 per_layer_projection_norm = 4.9B.
_force_cpu()    { case "$1" in google/embeddinggemma-2) echo "1"           ;; *) echo "0" ;; esac; }
# ─────────────────────────────────────────────────────────────────────────────

FIRST="${1:-}"
if [[ "$FIRST" == "--suite" ]]; then
  CORPUS="${2:-../ecfr-markdown/corpus}"
  OUT="${3:-build/embed}"
  RESULTS=()
  for m in "${SUITE[@]}"; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  SUITE: $m"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    tmpout=$(mktemp)
    # Run model; stream output live AND capture to tmpout for result grep.
    # The inner bash call exits non-zero on model failure; || true keeps the
    # suite running. With tee writing to both stdout and tmpout, live progress
    # appears in the log while we still have the output to grep.
    bash "$0" "$m" "$CORPUS" "$OUT" 2>&1 | tee "$tmpout" || true
    result=$(grep -E "^recall@[135]:" "$tmpout" | tr '\n' '  ' | sed 's/  $//' || true)
    [ -z "$result" ] && result="FAILED (see log above)"
    rm -f "$tmpout"
    RESULTS+=("$(printf '%-45s  %s' "$m" "$result")")
  done
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  SUITE RESULTS (dense + citation fast-path)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  for r in "${RESULTS[@]}"; do echo "  $r"; done
  echo ""
  exit 0
fi

MODEL_ID="${FIRST:?usage: $0 <hf-model-id>|--suite [corpus_dir] [out_dir]}"
CORPUS="${2:-../ecfr-markdown/corpus}"
OUT="${3:-build/embed}"
SLUG=$(echo "$MODEL_ID" | tr '[:upper:]' '[:lower:]' | tr '/-' '__')
DOC_PROMPT=$(_doc_prompt "$MODEL_ID")
QUERY_PROMPT=$(_query_prompt "$MODEL_ID")
MPS_BATCH=$(_mps_batch "$MODEL_ID")
FORCE_CPU=$(_force_cpu "$MODEL_ID")

mkdir -p "$OUT"

echo "==> Model: $MODEL_ID (slug: $SLUG)"
[[ -n "$DOC_PROMPT" ]] && echo "    doc_prompt=$DOC_PROMPT  query_prompt=$QUERY_PROMPT"
python3 -c "import sentence_transformers, torch, numpy, einops" 2>/dev/null || {
  echo "Installing dependencies..."
  pip install -q "sentence-transformers" "numpy" "einops" \
    --extra-index-url https://pypi.org/simple
}
# google/embeddinggemma-2 requires a recent transformers that registers
# the embedding_gemma2 architecture; upgrade unconditionally (no-op if current).
case "$MODEL_ID" in
  google/embeddinggemma-2)
    echo "==> Upgrading transformers for embeddinggemma-2 support..."
    pip install -q --upgrade transformers
    ;;
esac

echo "==> Building vectors"
python3 - "$CORPUS" "$OUT" "$MODEL_ID" "$SLUG" "$DOC_PROMPT" "$MPS_BATCH" "$FORCE_CPU" <<'PYEOF'
import sys
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

corpus_dir, out_dir, model_id, slug, doc_prompt, mps_batch, force_cpu = sys.argv[1:8]
mps_batch = int(mps_batch)
force_cpu = force_cpu == "1"

sys.path.insert(0, "build")
import importlib.util
spec = importlib.util.spec_from_file_location("bv", "build/build_vectors.py")
bv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bv)

corpus = Path(corpus_dir)
files = sorted(corpus.rglob("*.md"))
cur = [f for f in files if "--as-of-" not in f.name]
print(f"files: {len(cur)}", flush=True)

texts, paths = [], []
for f in cur:
    text = f.read_text(encoding="utf-8")
    fm, body = bv.parse_front_matter(text)
    title = fm.get("title", "")
    texts.append(f"{title}\n{body}".strip())
    paths.append(str(f.relative_to(corpus)))

if not force_cpu and torch.cuda.is_available():
    device, model_kwargs, batch_size = "cuda", {}, 32
elif not force_cpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device, model_kwargs, batch_size = "mps", {}, mps_batch
else:
    device, model_kwargs, batch_size = "cpu", {"torch_dtype": torch.float16}, 16

print(f"loading {model_id} ({device})...", flush=True)
m = SentenceTransformer(
    model_id,
    trust_remote_code=True,
    model_kwargs=model_kwargs,
    device=device,
)
print(f"dims: {m.get_sentence_embedding_dimension()}", flush=True)

encode_kw = {
    "batch_size": batch_size,
    "show_progress_bar": True,
    "normalize_embeddings": True,
}
if doc_prompt:
    encode_kw["prompt_name"] = doc_prompt

vecs = m.encode(texts, **encode_kw)
vecs = np.asarray(vecs, dtype=np.float32)
out = Path(out_dir) / f"vectors_{slug}.npz"
np.savez(out, vectors=vecs, paths=np.array(paths))
print(f"saved {out} {vecs.shape}", flush=True)
PYEOF

echo "==> Evaluating (dense + citation fast-path = free tier)"
EVAL_ARGS=(--vectors "$OUT/vectors_${SLUG}.npz" --cite --model "$MODEL_ID" --db data/meta.sqlite)
[[ -n "$QUERY_PROMPT" ]] && EVAL_ARGS+=(--query-prompt "$QUERY_PROMPT")
python3 golden/eval.py "${EVAL_ARGS[@]}"

echo "==> Done. Shipped baseline for comparison:"
echo "    python3 golden/eval.py --vectors data/vectors.npz --cite"
