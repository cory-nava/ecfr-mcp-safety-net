# Hosting the eCFR MCP Server

## Running Locally

### RAM Requirements

| Component | RAM |
|-----------|-----|
| `BAAI/bge-small-en-v1.5` model (CPU, PyTorch) | ~200 MB |
| Corpus data (vectors + chunks loaded into numpy) | ~100 MB |
| Python interpreter + MCP server overhead | ~100 MB |
| **Total** | **~400 – 512 MB** |

### Disk Requirements

| Component | Size |
|-----------|------|
| `BAAI/bge-small-en-v1.5` model weights (downloaded once to `~/.cache/huggingface`) | 127 MB |
| Corpus data files in `data/` (indexes, `meta.sqlite`, `corpus.tar.gz`) | ~DATA_MB MB |
| Python venv + dependencies (torch, sentence-transformers, mcp, etc.) | ~2–3 GB |
| **Total** | **~3.2 GB** |

### Startup Time

Startup loads the indexes (a second or two). The first search then loads the embedding model (downloading it once, ~127 MB, if it isn't cached yet), which takes 10–60 seconds depending on the machine and network; the first `get_section` reads the corpus tarball into memory (~1 s). After that, free-tier searches take well under a second; best-tier searches add the LLM rerank round trips (a few seconds).

---

## Cloud Hosting

### Why AWS Lambda is a Poor Fit

Lambda has three problems for this server:

1. **stdio transport** — the server runs via `mcp.run_stdio_async()`, a persistent process. Lambda is request/response and doesn't support persistent stdio. Switching to MCP's HTTP/SSE transport requires a code rewrite.
2. **Cold start latency** — importing PyTorch and loading the 127 MB model takes 10–30 seconds. Without Provisioned Concurrency, every cold start is painful.
3. **Package size** — the venv is 2–3 GB, requiring a container image build (Lambda supports up to 10 GB images, so it fits, but adds pipeline complexity).

Lambda only makes sense with Provisioned Concurrency to stay warm — at which point it costs the same as always-on alternatives with more setup work.

---

### Monthly Cost Estimates (us-east-1, as of Oct 2026)

| Option | Notes | Est. Cost/mo |
|--------|-------|-------------|
| **AWS Lambda** + Provisioned Concurrency | 1 instance × 1 GB × 720 hrs warm; requires HTTP transport rewrite + container image | ~$20–22 |
| **AWS ECS Fargate** | 0.5 vCPU, 2 GB RAM, always-on; runs existing stdio server unchanged in a container | ~$22–25 |
| **AWS EC2 t4g.small** | 2 GB RAM, ARM, always-on; cheapest AWS option, requires managing the instance | ~$12–14 |
| **Fly.io** | 1 shared CPU, 1–2 GB RAM machine; runs the container as-is, scales to zero (with cold start caveat) | ~$5–10 |
| **Railway** | Similar to Fly.io; simple Docker deploy | ~$5–10 |

### Recommendation

**ECS Fargate** is the cleanest AWS path — runs the existing server unchanged, stays warm, no transport rewrite needed. If cost matters more than staying in AWS, **Fly.io** or **Railway** cut the bill by ~60–70% with minimal extra setup.
