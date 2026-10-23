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
  "nomic-ai/nomic-embed-text-v1.5"   # competitive at bge-small size
  "Qwen/Qwen3-Embedding-0.6B"        # strong MTEB, larger download
  "google/embeddinggemma-2"          # Google DeepMind, 768d, MTEB 68.46
)

# Per-model prompt names for asymmetric embedding models.
# Using case statements instead of declare -A for bash 3.x compatibility (macOS default).
_doc_prompt()   { case "$1" in google/embeddinggemma-2) echo "Document"    ;; *) echo "" ;; esac; }
_query_prompt() { case "$1" in google/embeddinggemma-2) echo "SearchQuery" ;; *) echo "" ;; esac; }
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
    result=$(bash "$0" "$m" "$CORPUS" "$OUT" 2>&1 | tee /dev/stderr \
      | grep -E "^recall@[135]:" | tr '\n' '  ' | sed 's/  $//')
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

mkdir -p "$OUT"

echo "==> Model: $MODEL_ID (slug: $SLUG)"
[[ -n "$DOC_PROMPT" ]] && echo "    doc_prompt=$DOC_PROMPT  query_prompt=$QUERY_PROMPT"
python3 -c "import sentence_transformers, torch, numpy" 2>/dev/null || {
  echo "Installing sentence-transformers + torch (CPU)..."
  pip install -q "sentence-transformers" "numpy" \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple
}

echo "==> Building vectors"
python3 - "$CORPUS" "$OUT" "$MODEL_ID" "$SLUG" "$DOC_PROMPT" <<'PYEOF'
import sys
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

corpus_dir, out_dir, model_id, slug, doc_prompt = sys.argv[1:6]

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

use_cuda = torch.cuda.is_available()
print(f"loading {model_id} ({'cuda' if use_cuda else 'cpu, fp16'})...", flush=True)
m = SentenceTransformer(
    model_id,
    trust_remote_code=True,
    model_kwargs={"torch_dtype": torch.float16} if not use_cuda else {},
    device="cuda" if use_cuda else "cpu",
)
print(f"dims: {m.get_sentence_embedding_dimension()}", flush=True)

encode_kw = {
    "batch_size": 32 if use_cuda else 16,
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
EVAL_ARGS=(--vectors "$OUT/vectors_${SLUG}.npz" --cite --model "$MODEL_ID")
[[ -n "$QUERY_PROMPT" ]] && EVAL_ARGS+=(--query-prompt "$QUERY_PROMPT")
python3 golden/eval.py "${EVAL_ARGS[@]}"

echo "==> Done. Shipped baseline for comparison:"
echo "    python3 golden/eval.py --vectors data/vectors.npz --cite"
