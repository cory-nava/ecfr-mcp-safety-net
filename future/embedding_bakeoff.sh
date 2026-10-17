#!/usr/bin/env bash
# Generic embedding-model bake-off for ecfr-mcp.
#
# Builds section vectors with any sentence-transformers-compatible embedding
# model and evaluates dense retrieval on the frozen golden set, for comparison
# against the shipped bge-small-en-v1.5 index.
#
# Usage:
#   bash future/embedding_bakeoff.sh <hf-model-id> [corpus_dir] [out_dir]
#
# Examples:
#   bash future/embedding_bakeoff.sh BAAI/bge-large-en-v1.5
#   bash future/embedding_bakeoff.sh Qwen/Qwen3-Embedding-0.6B
#   bash future/embedding_bakeoff.sh nomic-ai/nomic-embed-text-v1.5
#   bash future/embedding_bakeoff.sh jinaai/jina-embeddings-v3
#
# REQUIREMENTS: RAM scales with model size. bge-large (335M) and
# Qwen3-Embedding-0.6B fit in ~8GB in fp16; larger models need more.
# No API spend: dense eval only.
#
# Output:
#   <out_dir>/vectors_<slug>.npz   section vectors (vectors, paths)
#   stdout: recall@1/3/5 overall and per query type

set -euo pipefail

MODEL_ID="${1:?usage: $0 <hf-model-id> [corpus_dir] [out_dir]}"
CORPUS="${2:-../ecfr-markdown/corpus}"
OUT="${3:-build/embed}"
SLUG=$(echo "$MODEL_ID" | tr '[:upper:]' '[:lower:]' | tr '/-' '__')

mkdir -p "$OUT"

echo "==> Model: $MODEL_ID (slug: $SLUG)"
python3 -c "import sentence_transformers, torch, numpy" 2>/dev/null || {
  echo "Installing sentence-transformers + torch (CPU)..."
  pip install -q "sentence-transformers" "numpy" \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple
}

echo "==> Building vectors"
python3 - "$CORPUS" "$OUT" "$MODEL_ID" "$SLUG" <<'PYEOF'
import sys
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

corpus_dir, out_dir, model_id, slug = sys.argv[1:5]

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
vecs = m.encode(
    texts,
    batch_size=32 if use_cuda else 16,
    show_progress_bar=True,
    normalize_embeddings=True,
)
vecs = np.asarray(vecs, dtype=np.float32)
out = Path(out_dir) / f"vectors_{slug}.npz"
np.savez(out, vectors=vecs, paths=np.array(paths))
print(f"saved {out} {vecs.shape}", flush=True)
PYEOF

echo "==> Evaluating (dense + citation fast-path = free tier)"
python3 golden/eval.py --vectors "$OUT/vectors_${SLUG}.npz" --cite \
  --model "$MODEL_ID"

echo "==> Done. Shipped baseline for comparison:"
echo "    python3 golden/eval.py --vectors data/vectors.npz --cite"
