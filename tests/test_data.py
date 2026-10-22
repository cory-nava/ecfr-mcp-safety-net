"""Integrity checks on the shipped indexes and the lookup tools. No embedding model needed."""
import json
import sqlite3
import tarfile
import unittest
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path

from ecfr_mcp import retrieval

DATA = Path(__file__).parent.parent / "data"


class IndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = retrieval.Corpus(DATA)
        cls.con = sqlite3.connect(DATA / "meta.sqlite")

    @classmethod
    def tearDownClass(cls):
        cls.con.close()

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

    def test_one_current_version_and_ordered_history(self):
        cites = [c for (c,) in self.con.execute(
            "SELECT citation FROM docs GROUP BY citation HAVING COUNT(*) > 1")]
        self.assertTrue(cites)
        for c in cites:
            hist = retrieval.version_history(DATA, c)
            self.assertLessEqual(sum(h["is_current"] for h in hist), 1, c)
            # versions never overlap; a gap is legitimate (e.g. 42 CFR 457.960 was removed 2024-06-03 to 2026-06-02)
            for a, b in zip(hist, hist[1:]):
                if a["effective_to"] and b["effective_from"]:
                    self.assertLess(a["effective_to"], b["effective_from"], c)


class LookupTests(unittest.TestCase):
    def setUp(self):
        # an earlier version with a known last day, followed the next day by another version
        with closing(sqlite3.connect(DATA / "meta.sqlite")) as con:
            self.cite, self.last_day = con.execute(
                "SELECT a.citation, a.effective_to FROM docs a JOIN docs b ON a.citation = b.citation "
                "AND b.effective_from = date(a.effective_to, '+1 day') "
                "WHERE a.is_current = 0 ORDER BY a.citation LIMIT 1").fetchone()

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

    def _one(self, sql):
        with closing(sqlite3.connect(DATA / "meta.sqlite")) as con:
            return con.execute(sql).fetchone()

    def test_as_of_before_recorded_history_is_not_called_repealed(self):
        cite, earliest = self._one(
            "SELECT citation, MIN(effective_from) FROM docs WHERE citation IN "
            "(SELECT citation FROM docs WHERE is_current = 1) GROUP BY citation ORDER BY citation LIMIT 1")
        day_before = (date.fromisoformat(earliest) - timedelta(days=1)).isoformat()
        sec = retrieval.get_section(DATA, cite, as_of=day_before)
        self.assertFalse(sec["found"])
        self.assertEqual(sec["status"], "before_recorded_history")
        self.assertEqual(sec["earliest_effective_from"], earliest)
        self.assertNotIn("repealed", sec["note"])

    def test_as_of_in_a_gap_reports_the_gap(self):
        # a version whose successor starts later than the day after it ends (e.g. 42 CFR 457.960)
        cite, ended, starts = self._one(
            "SELECT a.citation, a.effective_to, MIN(b.effective_from) FROM docs a JOIN docs b "
            "ON a.citation = b.citation AND b.effective_from > a.effective_to "
            "WHERE a.effective_to != '' GROUP BY a.citation, a.effective_to "
            "HAVING MIN(b.effective_from) > date(a.effective_to, '+1 day') ORDER BY a.citation LIMIT 1")
        inside = (date.fromisoformat(ended) + timedelta(days=1)).isoformat()
        sec = retrieval.get_section(DATA, cite, as_of=inside)
        self.assertEqual(sec["status"], "not_in_force_on_date")
        self.assertEqual((sec["gap_from"], sec["gap_to"]), (ended, starts))
        self.assertNotIn("repealed", sec["note"])

    def test_no_current_text_is_not_in_force(self):
        cite, last_to = self._one(
            "SELECT citation, MAX(effective_to) FROM docs GROUP BY citation "
            "HAVING SUM(is_current) = 0 ORDER BY citation LIMIT 1")
        for as_of in (None, (date.fromisoformat(last_to) + timedelta(days=1)).isoformat()):
            sec = retrieval.get_section(DATA, cite, as_of=as_of)
            self.assertEqual(sec["status"], "not_in_force", as_of)
            self.assertEqual(sec["last_effective_to"], last_to)
        self.assertTrue(retrieval.get_section(DATA, cite, as_of=last_to)["found"])

    def test_malformed_as_of_is_rejected_not_guessed(self):
        for bad in ("2024-6-1", "June 2024", "2024-02-30", "2025-01-17T00:00", "06/01/2024"):
            sec = retrieval.get_section(DATA, "7 CFR 273.9", as_of=bad)
            self.assertFalse(sec["found"], bad)
            self.assertEqual(sec["status"], "invalid_date", bad)
        self.assertEqual(retrieval.get_section(DATA, "7 CFR 999.999")["status"], "unknown_citation")

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
