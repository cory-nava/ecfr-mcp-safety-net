"""Best-tier tests with a stubbed LLM: scoring path, graceful degradation, citation bypass.

The query embedder is stubbed with a vector taken from the index itself, so no model
download is needed; test_data.py checks that the indexes and the model agree on dimensions.
"""
import json
import os
import re
import time
import unittest
from pathlib import Path
from unittest import mock

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



class _Resp:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class LocalLLMTests(unittest.TestCase):
    def test_local_url_enables_best_tier_without_key(self):
        tier, reason = retrieval.resolve_tier({"ECFR_MCP_LLM_URL": "http://localhost:1234/v1/"})
        self.assertEqual(tier, "best")
        self.assertIn("http://localhost:1234/v1,", reason)
        self.assertIn(retrieval.LOCAL_MODEL, reason)
        self.assertEqual(retrieval._rerank_model({"ECFR_MCP_LLM_URL": "x", "ECFR_MCP_LLM_MODEL": "qwen"}), "qwen")

    def test_chat_posts_to_local_server(self):
        sent = []

        def fake_urlopen(req, timeout):
            sent.append((req, json.loads(req.data)))
            return _Resp({"choices": [{"message": {"content": '{"1": 7}'}}], "usage": {}})

        env = {"ECFR_MCP_LLM_URL": "http://localhost:1234/v1"}
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(retrieval.urllib.request, "urlopen", fake_urlopen):
            text, _ = retrieval._chat("hi", "google/gemma-4-12b", None, 0.0, 500, 180)
        req, body = sent[0]
        self.assertEqual(text, '{"1": 7}')
        self.assertEqual(req.full_url, "http://localhost:1234/v1/chat/completions")
        self.assertNotIn("Authorization", req.headers)
        self.assertEqual(body["reasoning_effort"], "none")
        self.assertEqual(body["max_tokens"], 500)

    def test_local_batches_run_one_at_a_time(self):
        import threading
        lock, active, peak = threading.Lock(), [0], [0]

        def fake_api(prompt: str):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.01)
            with lock:
                active[0] -= 1
            return "{}", {}

        pairs = [("q", f"passage {i}") for i in range(retrieval.TOP_N)]
        with mock.patch.dict(os.environ, {"ECFR_MCP_LLM_URL": "http://localhost:1234/v1"}):
            out = retrieval._llm_scores(pairs, None, call_api=fake_api)
        self.assertEqual(len(out), retrieval.TOP_N // retrieval.BATCH)
        self.assertEqual(peak[0], 1)

    def test_local_failure_never_falls_back_to_cloud(self):
        corpus = retrieval.Corpus(DATA)
        q = corpus.sec_vecs[0]
        corpus.embed = lambda text: q
        urls = []

        def refuse(req, timeout):
            urls.append(req.full_url)
            raise OSError("connection refused")

        env = {"ECFR_MCP_LLM_URL": "http://localhost:1234/v1", "OPENROUTER_API_KEY": "sk-test"}
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(retrieval.urllib.request, "urlopen", refuse), \
                mock.patch.object(retrieval.time, "sleep", lambda s: None):
            hits, degraded = retrieval.search_best(corpus, QUERY, top_k=3)
        self.assertTrue(degraded and hits)
        self.assertTrue(urls)
        self.assertTrue(all(u.startswith("http://localhost:1234/") for u in urls))


if __name__ == "__main__":
    unittest.main()
