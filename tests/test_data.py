"""Integrity checks on the shipped indexes and the lookup tools. No embedding model needed."""
import json
import sqlite3
import tarfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from ecfr_mcp import retrieval

DATA = Path(__file__).parent.parent / "data"


class IndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = retrieval.Corpus(DATA)
        cls.con = sqlite3.connect(DATA / "meta.sqlite")

    def test_indexes_share_one_model(self):
        manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(self.corpus.sec_vecs.shape[1], manifest["dims"])
        self.assertEqual(self.corpus.chunk_vecs.shape[1], manifest["dims"])
        self.assertEqual(manifest.get("chunks_model", manifest["model"]), manifest["model"])

    def test_meta_rows_unique_and_typed(self):
        n, distinct = self.con.execute("SELECT COUNT(*), COUNT(DISTINCT path) FROM docs").fetchone()
        self.assertEqual(n, distinct)
        types = {t for (t,) in self.con.execute("SELECT DISTINCT typeof(is_current) FROM docs")}
        self.assertEqual(types, {"integer"})

    def test_every_index_entry_resolves(self):
        current = {p for (p,) in self.con.execute("SELECT path FROM docs WHERE is_current=1")}
        self.assertEqual(set(self.corpus.sections), current)
        self.assertLessEqual(set(self.corpus.chunk_sections), current)
        with tarfile.open(DATA / "corpus.tar.gz") as tf:
            names = {m.name[len("corpus/"):] for m in tf if m.isfile()}
        missing = {p for (p,) in self.con.execute("SELECT path FROM docs")} - names
        self.assertFalse(missing, f"{len(missing)} meta rows have no file in corpus.tar.gz")

    def test_one_current_version_and_contiguous_history(self):
        cites = [c for (c,) in self.con.execute(
            "SELECT citation FROM docs GROUP BY citation HAVING COUNT(*) > 1")]
        self.assertTrue(cites)
        for c in cites:
            hist = retrieval.version_history(DATA, c)
            self.assertLessEqual(sum(h["is_current"] for h in hist), 1, c)
            for a, b in zip(hist, hist[1:]):
                if a["effective_to"] and b["effective_from"]:
                    nxt = (date.fromisoformat(a["effective_to"]) + timedelta(days=1)).isoformat()
                    self.assertEqual(nxt, b["effective_from"], c)


class LookupTests(unittest.TestCase):
    def setUp(self):
        con = sqlite3.connect(DATA / "meta.sqlite")
        # an earlier version with a known last day, and the version that follows it
        self.cite, self.last_day = con.execute(
            "SELECT citation, effective_to FROM docs WHERE is_current=0 AND effective_to != '' "
            "ORDER BY citation LIMIT 1").fetchone()

    def test_as_of_includes_last_day_in_force(self):
        sec = retrieval.get_section(DATA, self.cite, as_of=self.last_day)
        self.assertTrue(sec["found"])
        self.assertEqual(sec["effective_to"], self.last_day)
        self.assertFalse(sec["is_current"])
        self.assertTrue(sec["text"].startswith("---\n"))
        nxt = (date.fromisoformat(self.last_day) + timedelta(days=1)).isoformat()
        after = retrieval.get_section(DATA, self.cite, as_of=nxt)
        self.assertTrue(after["found"])
        self.assertEqual(after["effective_from"], nxt)

    def test_current_section_and_missing(self):
        sec = retrieval.get_section(DATA, "7  CFR 273.9")
        self.assertTrue(sec["found"] and sec["is_current"])
        self.assertIn("income", sec["text"].lower())
        self.assertFalse(retrieval.get_section(DATA, "7 CFR 999.999")["found"])

    def test_list_programs(self):
        progs = retrieval.list_programs(DATA)
        self.assertGreaterEqual(len(progs), 10)
        self.assertTrue(all(isinstance(p["current_sections"], int) for p in progs))


class TierTests(unittest.TestCase):
    def test_resolve_tier(self):
        self.assertEqual(retrieval.resolve_tier({})[0], "free")
        self.assertEqual(retrieval.resolve_tier({"OPENROUTER_API_KEY": "k"})[0], "best")
        self.assertEqual(retrieval.resolve_tier({"OPENROUTER_API_KEY": "k", "ECFR_MCP_TIER": "free"})[0], "free")


if __name__ == "__main__":
    unittest.main()
