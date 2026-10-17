# Golden eval set

50 hand-authored queries over the ecfr-markdown corpus (snapshot 2026-10-11,
eCFR current as of 2026-10-08), each mapped to one expected section citation.

## Method

- **Frozen before any retrieval was built or run** (2026-10-12). No query was
  tuned against embedding output. This is the seal: the set measures the
  retriever, not the author's ability to fit it.
- **Stratified** across 14 program areas (SNAP 14, SSI 8, Medicaid/CHIP 5,
  housing 4, TANF 3, others 1-2 each).
- **Three query types:** `exact` (5; the citation itself, e.g. "7 CFR 273.9"),
  `natural` (33; plain questions derived from section titles), `paraphrase`
  (12; caseworker-style plain language, e.g. "how much can I earn and still
  get food stamps"). The paraphrase slice is the hard one and the honest
  measure of semantic retrieval.
- Every `expected_citation` was verified to exist as a **current-version**
  file in the corpus manifest. Queries whose sections had no current file
  were replaced (see data-quality notes).

## Running

```bash
# what ships: the server's own retrieval code over data/ (citation fast-path included)
python golden/eval_server.py --tier free
OPENROUTER_API_KEY=... python golden/eval_server.py --tier best

# vectors only (no citation fast-path), with the model named in data/manifest.json
python golden/eval.py --vectors data/vectors.npz
```

Reports recall@1 / recall@3 / recall@5 overall and per query type. A query
counts as correct if the expected citation's file is in the top-k.

The `exp_*.py` and `eval_*.py` experiment scripts are the record of the
2026-10-12/13 bake-off (see `results-2026-10-12.md` and `../DECISION_LOG.md`).
They were run with `BAAI/bge-base-en-v1.5`, which is still their default
`--model`; the shipped indexes now use `bge-small`. The LLM experiments
read `OPENROUTER_API_KEY`.

## Updating

- Add queries; never edit an existing query to make the retriever pass.
- If the corpus snapshot changes, re-verify every `expected_path` against the
  new manifest before running.
- Paraphrase queries are the most valuable additions: they are the closest
  proxy for real agent/caseworker usage.

## Data-quality notes (found while authoring, 2026-10-12)

Two sections have version-history files but **no current-version file** in the
corpus, i.e. they appear in no current part XML export:

- 45 CFR 261.30 (TANF work activities) — only `--as-of-2024-01-01` and
  `--as-of-2026-07-31` versions exist
- 45 CFR 96.80 (LIHEAP scope) — only `--as-of-2024-01-01` and
  `--as-of-2026-06-09` versions exist

Resolved 2026-10-17: not an export bug. Both sections became `[Reserved]`
(45 CFR 261.30 on 2026-09-29, 45 CFR 96.80 on 2026-08-10), and ecfr-markdown
skips reserved sections, so only their earlier versions exist, each with an
`effective_to`.
