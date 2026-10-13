"""Best-tier tests with stubbed LLM: scoring path + graceful degradation."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import retrieval

DATA = Path(__file__).parent.parent / "data"


def test_rerank_uses_llm_scores():
    corpus = retrieval.Corpus(DATA)

    def fake_api(prompt: str):
        # score every passage 0 except ones mentioning 273.9 -> 10
        # (note: tokenizer-decoded chunk texts render it as "273. 9")
        scores = {}
        for line in prompt.split("\n"):
            m = re.match(r"\[(\d+)\] (.*)", line)
            if m:
                scores[m.group(1)] = 10 if "273." in m.group(2) else 0
        return json.dumps(scores), {}

    hits, degraded = retrieval.search_best(
        corpus, "SNAP income rules", top_k=3, api_key="stub", call_api=fake_api)
    assert not degraded
    assert hits, "no hits"
    print("rerank top3:", [(h.citation, h.score) for h in hits])
    assert hits[0].score == 10


def test_degraded_falls_back_to_free():
    corpus = retrieval.Corpus(DATA)

    def boom(prompt: str):
        raise RuntimeError("provider exploded")

    hits, degraded = retrieval.search_best(
        corpus, "SNAP income rules", top_k=3, api_key="stub", call_api=boom)
    assert degraded, "should report degradation"
    free_hits = retrieval.search_free(corpus, "SNAP income rules", top_k=3)
    assert [h.citation for h in hits] == [h.citation for h in free_hits]
    print("degradation OK: fell back to", [h.citation for h in hits])


def test_citation_bypass_in_best_tier():
    corpus = retrieval.Corpus(DATA)
    calls = []

    def spy(prompt: str):
        calls.append(prompt)
        return "{}", {}

    hits, degraded = retrieval.search_best(
        corpus, "7 CFR 273.9", top_k=3, api_key="stub", call_api=spy)
    assert hits[0].citation == "7 CFR 273.9" and not calls, \
        "citation should bypass the LLM entirely"
    print("citation bypass OK")


if __name__ == "__main__":
    test_rerank_uses_llm_scores()
    test_degraded_falls_back_to_free()
    test_citation_bypass_in_best_tier()
    print("BEST-TIER TESTS OK")
