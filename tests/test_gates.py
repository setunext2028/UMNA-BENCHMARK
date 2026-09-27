import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class TestGates(unittest.TestCase):
    def test_lint_human_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "tools/lint_human.py"],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode, 0,
            f"lint_human failed:\n{result.stdout}\n{result.stderr}",
        )

    @unittest.skipIf(
        os.environ.get("UMNA_AUDIT_RUNNING") == "1",
        "audit already runs the suite when it is the caller",
    )
    def test_audit_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "tools/audit.py"],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode, 0,
            f"audit failed:\n{result.stdout}\n{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()