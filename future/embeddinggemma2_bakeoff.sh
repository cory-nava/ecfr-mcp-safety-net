#!/usr/bin/env bash
# EmbeddingGemma 2 bake-off for ecfr-mcp.
#
# Builds section vectors with google/embeddinggemma-2 (768-dim, text-only)
# and evaluates dense retrieval on the frozen golden set, for comparison
# against the shipped bge-small-en-v1.5 index.
#
# REQUIREMENTS: a box with 16GB+ RAM (the 740M multimodal checkpoint OOMs
# on 7GB even in fp16). CPU works; CUDA is faster if available.
# No API spend: dense eval only. The rerank comparison needs OpenRouter.
#
# Usage:
#   bash future/embeddinggemma2_bakeoff.sh [corpus_dir] [out_dir]
#
# Defaults: corpus_dir=../ecfr-markdown/corpus (sibling checkout),
#           out_dir=build/eg2
#
# Output:
#   <out_dir>/vectors_eg2.npz   section vectors (vectors, paths)
#   stdout: recall@1/3/5 overall and per query type
#
# To compare reranked quality, point the rerank experiment at the new
# vectors afterwards (needs OPENROUTER_API_KEY):
#   python golden/exp_rerank_candidates.py --step generate \
#     --candidate-source hyde_avg --model-llm <model> \
#     --chunks data/chunks.npz --meta data/meta.sqlite \
#     --model-emb google/embeddinggemma-2 \
#     --scores golden/llm_rerank_scores_eg2.jsonl
# (Note: chunk vectors also need rebuilding with eg2 for a full rerank
# comparison; see build/build_chunks.py.)

set -euo pipefail

CORPUS="${1:-../ecfr-markdown/corpus}"
OUT="${2:-build/eg2}"
MODEL="google/embeddinggemma-2"

mkdir -p "$OUT"

echo "==> Checking dependencies"
python3 -c "import sentence_transformers, torch, numpy" 2>/dev/null || {
  echo "Installing sentence-transformers + torch (CPU)..."
  pip install -q "sentence-transformers" "numpy" \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple
}
python3 -c "import torchvision" 2>/dev/null || {
  echo "Installing torchvision (needed by the EmbeddingGemma2 processor)..."
  pip install -q torchvision \
    --index-url https://download.pytorch.org/whl/cpu
}

echo "==> Building vectors with $MODEL"
echo "    corpus: $CORPUS"
echo "    out:    $OUT/vectors_eg2.npz"

python3 - "$CORPUS" "$OUT" "$MODEL" <<'PYEOF'
import sys
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

corpus_dir, out_dir, model_id = sys.argv[1], sys.argv[2], sys.argv[3]

# --- read corpus (same logic as build/build_vectors.py) ---
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

# --- embed ---
use_cuda = torch.cuda.is_available()
print(f"loading {model_id} ({'cuda' if use_cuda else 'cpu, fp16'})...", flush=True)
m = SentenceTransformer(
    model_id,
    trust_remote_code=True,
    model_kwargs={"torch_dtype": torch.float16} if not use_cuda else {},
    device="cuda" if use_cuda else "cpu",
)
print(f"dims: {m.get_sentence_embedding_dimension()}", flush=True)
vecs = m.encode(
    texts,
    batch_size=32 if use_cuda else 8,
    show_progress_bar=True,
    normalize_embeddings=True,
)
vecs = np.asarray(vecs, dtype=np.float32)
print("vecs:", vecs.shape, flush=True)
np.savez(Path(out_dir) / "vectors_eg2.npz",
         vectors=vecs, paths=np.array(paths))
print("saved", flush=True)
PYEOF

echo "==> Evaluating on the golden set (dense + citation fast-path = free tier)"
python3 golden/eval.py --vectors "$OUT/vectors_eg2.npz" --cite \
  --model "$MODEL"

echo "==> Done. Compare against shipped bge-small-en-v1.5:"
echo "    python3 golden/eval.py --vectors data/vectors.npz --cite"
