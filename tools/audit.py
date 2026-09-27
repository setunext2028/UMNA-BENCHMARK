"""Mechanical self-audit checks from docs/07.

Each check below can fail. Checks that require judgment (the hand-traced
paths, the consumer hunt, and the claim check) remain printed reminders.

Usage: python tools/audit.py   (exits non-zero on failure)
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from runtime.registry import ManifestError, load_manifest

FAILURES: list[str] = []


def check(flag, success, failure=None):
    if flag:
        print(f"ok    {success}")
        return
    message = failure or success
    FAILURES.append(message)
    print(f"FAIL  {message}")


def _tree(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports_name(path, name):
    for node in ast.walk(_tree(path)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if any(alias.name == name for alias in node.names):
                return True
    return False


def _imports_module(path, module):
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            if any(
                alias.name == module or alias.name.startswith(f"{module}.")
                for alias in node.names
            ):
                return True
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == module or node.module.startswith(f"{module}."):
                return True
    return False


def check_manifest_allow_list():
    """B5: every on-disk manifest passes the actual strict loader."""

    for path in sorted((REPO / "wings").glob("*/manifest.yaml")):
        try:
            load_manifest(path)
        except ManifestError as error:
            check(
                False,
                "",
                f"{path.relative_to(REPO)} failed strict validation: {error}",
            )
        else:
            check(True, f"{path.relative_to(REPO)} passes strict validation")


def check_planner_reads_no_manifest():
    """B5: the planner cannot import Manifest and read vendor metadata."""

    for path in sorted((REPO / "planner").rglob("*.py")):
        check(
            not _imports_name(path, "Manifest"),
            f"{path.relative_to(REPO)} does not import Manifest",
            f"{path.relative_to(REPO)} imports Manifest; routing numbers "
            "must come from HostMetrics",
        )


def check_execute_has_no_rogue_callers():
    """B6: no current code path imports run directly from runtime.execute."""

    roots = ["runtime", "planner", "risk", "wings", "core", "bench"]
    for root in roots:
        for path in sorted((REPO / root).rglob("*.py")):
            if path == REPO / "runtime" / "execute.py":
                continue
            tree = _tree(path)
            direct_import = any(
                isinstance(node, ast.ImportFrom)
                and node.module == "runtime.execute"
                and any(alias.name == "run" for alias in node.names)
                for node in ast.walk(tree)
            )
            direct_call = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "run"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "execute"
                for node in ast.walk(tree)
            )
            check(
                not direct_import and not direct_call,
                f"{path.relative_to(REPO)} has no direct execute.run path",
                f"{path.relative_to(REPO)} reaches execute.run directly; "
                "the path must pass admission",
            )


def check_layer_boundaries():
    """runtime never imports planner/wings; planner never imports wings."""

    for path in sorted((REPO / "runtime").rglob("*.py")):
        check(
            not _imports_module(path, "planner")
            and not _imports_module(path, "wings"),
            f"{path.relative_to(REPO)} respects runtime dependency direction",
            f"{path.relative_to(REPO)} imports planner/ or wings/",
        )
    for path in sorted((REPO / "planner").rglob("*.py")):
        check(
            not _imports_module(path, "wings"),
            f"{path.relative_to(REPO)} plans over refs, not Wing objects",
            f"{path.relative_to(REPO)} imports wings/",
        )


def check_trace_terms():
    """B7: latency has exactly the eight terms in docs/05."""

    tree = _tree(REPO / "runtime" / "execute.py")
    fields = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "LatencyBreakdown":
            for statement in node.body:
                if (
                    isinstance(statement, ast.AnnAssign)
                    and isinstance(statement.target, ast.Name)
                ):
                    fields.add(statement.target.id)
    expected = {
        "retrieve_ms",
        "plan_ms",
        "admit_ms",
        "load_ms",
        "transfer_ms",
        "wing_ms",
        "fusion_ms",
        "risk_ms",
    }
    check(
        fields == expected,
        "LatencyBreakdown carries all eight terms",
        f"LatencyBreakdown terms {sorted(fields)}; expected {sorted(expected)}",
    )


def check_version_pinning():
    """Every declared package version remains 0.1.0."""

    for path in sorted(REPO.rglob("__init__.py")):
        text = path.read_text(encoding="utf-8")
        if "VERSION" not in text:
            continue
        check(
            'VERSION = "0.1.0"' in text,
            f"{path.relative_to(REPO)} keeps version 0.1.0",
            f"{path.relative_to(REPO)} declares a version other than 0.1.0",
        )


def run_tests():
    """The test suite must pass; a red suite is a red audit."""

    env = dict(os.environ)
    env["UMNA_AUDIT_RUNNING"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=env,
    )
    check(
        result.returncode == 0,
        "test suite passes",
        "test suite failed; run it directly for the traceback",
    )


def main():
    print("tools/audit.py - self-audit (docs/07 A and B, mechanical)")
    check_manifest_allow_list()
    check_planner_reads_no_manifest()
    check_execute_has_no_rogue_callers()
    check_layer_boundaries()
    check_trace_terms()
    check_version_pinning()
    run_tests()
    if FAILURES:
        print(f"\n{len(FAILURES)} failure(s)")
        sys.exit(1)
    print("\nall mechanical checks pass")
    print("reminder: hand-trace one happy path and one failure path;")
    print("         find the consumer of anything new; classify claims")


if __name__ == "__main__":
    main()
