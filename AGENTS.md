# Notes for coding agents

ecfr-mcp is a stdio MCP server over the ctrimm/ecfr-markdown-safety-net corpus. `ecfr_mcp/` is the server
(`server.py` tools, `retrieval.py` tiers and lookups), `data/` the shipped indexes and corpus, `build/` the
index builders, `golden/` the eval set.

## Rules

- **`data/manifest.json` is the source of truth for the embedding model.** The server loads the model it names
  and refuses to start if `vectors.npz`, `chunks.npz` and the manifest disagree on dimensions.
- **Rebuild `data/` as a unit, in order**, never one file by hand:

  ```bash
  python build/build_vectors.py --corpus ../ecfr-markdown-safety-net/corpus --out data \
      --snapshot "ctrimm/ecfr-markdown-safety-net @ <commit> (eCFR current as of <date>)"
  python build/build_chunks.py  --corpus ../ecfr-markdown-safety-net/corpus --out data
  python build/pack_corpus.py   --corpus ../ecfr-markdown-safety-net/corpus --out data
  ```

  (Switching models once updated only `vectors.npz`; the best tier then failed on every query, silently.)
- **The degrade-to-free fallback is for LLM failures only.** Don't widen the `except` in `search_best` or move
  embedding inside it; hidden errors are how that failure went unnoticed.
- **The server writes nothing to disk.** `meta.sqlite` opens read-only and section text is read from
  `corpus.tar.gz` into memory. Keep it working on Windows (no `fcntl`, no POSIX-only paths).
- **Dates.** `effective_to` is the last day a version was in force, inclusive; `as_of` lookups use `>=`.
  Versions never overlap, but gaps are real (removed sections). Dates compare as strings, so `as_of` must be
  validated as `YYYY-MM-DD` before any query. A miss must say which case it is (gap, before recorded history,
  no current text); never report a section as repealed unless it has no current version.
- **`is_current` is an integer** in `meta.sqlite`; report it with `bool()` only after that.

## Changing retrieval

1. Measure before and after on the frozen golden set:
   `python golden/eval_server.py --tier free` (the shipped code), or
   `python golden/eval.py --vectors <dir>/vectors.npz --cite` for an index outside `data/`.
2. Record the numbers in a dated `golden/results-YYYY-MM-DD.md` and the decision in `DECISION_LOG.md`.
3. Update the tier table in `README.md`.

Add golden queries freely; never edit an existing query to make a retriever pass. The best tier needs
`OPENROUTER_API_KEY` to measure; don't publish best-tier numbers you didn't measure.

## Checks

```bash
python -m unittest discover -s tests -t .
```

CI installs only `numpy` and `mcp` (Ubuntu and Windows, Python 3.10 and 3.13). `tests/test_data.py` checks index
integrity and lookups, `tests/test_best_tier.py` stubs the LLM and the embedder, and `tests/test_smoke.py` drives all
four tools over stdio; it needs `sentence-transformers` and skips without it.

## Don'ts

- No hard-coded machine paths, credentials or sandbox workarounds; experiment scripts read `OPENROUTER_API_KEY`.
- Don't commit build scratch: logs, `build/*.npz`, `*.spent`, an extracted `data/corpus/` (see `.gitignore`).
