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
python3 golden/eval.py --vectors build/vectors.npz --meta build/meta.sqlite
```

Reports recall@1 / recall@3 / recall@5 overall and per query type. A query
counts as correct if the expected citation's file is in the top-k by cosine
similarity.

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

Either the sections were removed/renumbered after those dates, or the export
dropped them. Worth investigating in the ecfr-markdown repo before treating
the corpus as complete.
