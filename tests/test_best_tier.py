"""Best-tier tests with a stubbed LLM: scoring path, graceful degradation, citation bypass.

The query embedder is stubbed with a vector taken from the index itself, so no model
download is needed; test_data.py checks that the indexes and the model agree on dimensions.
"""
import json
import re
import unittest
from pathlib import Path

from ecfr_mcp import retrieval

DATA = Path(__file__).parent.parent / "data"
QUERY = "SNAP income rules"


class BestTierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = retrieval.Corpus(DATA)
        q = cls.corpus.sec_vecs[cls.corpus.sections.index("title-7/part-273/7-cfr-273-9.md")]
        cls.corpus.embed = lambda text: q

    def test_rerank_uses_llm_scores(self):
        def fake_api(prompt: str):
            # 10 for passages from part 273 (tokenizer-decoded text renders "273.9" as "273. 9"), else 0
            scores = {m.group(1): 10 if "273." in m.group(2) else 0
                      for m in re.finditer(r"^\[(\d+)\] (.*)$", prompt, re.MULTILINE)}
            return json.dumps(scores), {}

        hits, degraded = retrieval.search_best(self.corpus, QUERY, top_k=3, api_key="stub", call_api=fake_api)
        self.assertFalse(degraded)
        self.assertTrue(hits)
        self.assertEqual(hits[0].score, 10)

    def test_llm_failure_falls_back_to_free(self):
        def boom(prompt: str):
            raise RuntimeError("provider exploded")

        hits, degraded = retrieval.search_best(self.corpus, QUERY, top_k=3, api_key="stub", call_api=boom)
        self.assertTrue(degraded)
        free = retrieval.search_free(self.corpus, QUERY, top_k=3)
        self.assertEqual([h.citation for h in hits], [h.citation for h in free])

    def test_citation_bypasses_llm(self):
        calls = []
        hits, degraded = retrieval.search_best(self.corpus, "7 CFR 273.9", top_k=3, api_key="stub",
                                               call_api=lambda p: calls.append(p) or ("{}", {}))
        self.assertEqual(hits[0].citation, "7 CFR 273.9")
        self.assertFalse(calls or degraded)

    def test_embedding_errors_are_not_hidden(self):
        corpus = retrieval.Corpus(DATA)
        corpus.embed = lambda text: (_ for _ in ()).throw(ValueError("dimension mismatch"))
        with self.assertRaises(ValueError):
            retrieval.search_best(corpus, QUERY, api_key="stub", call_api=lambda p: ("{}", {}))


if __name__ == "__main__":
    unittest.main()
