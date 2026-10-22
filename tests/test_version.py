"""The package version and pyproject.toml agree (they drifted apart before: 0.1.0 vs 1.0.0)."""
import re
import unittest
from pathlib import Path

import ecfr_mcp


class VersionTests(unittest.TestCase):
    def test_package_version_matches_pyproject(self):
        pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text(encoding="utf-8")
        declared = re.search(r'(?m)^version = "([^"]+)"', pyproject).group(1)
        self.assertEqual(ecfr_mcp.__version__, declared)


if __name__ == "__main__":
    unittest.main()
