"""Style checks from docs/09_HUMAN_CODE.md.

The forbidden-vocabulary list is read from docs/09_HUMAN_CODE.md itself,
so the spec is the single source of truth and this file never spells the
words it enforces. Emoji, attribution trailers, changelog headers and
banner comments are matched by pattern.

Exempts docs/09_HUMAN_CODE.md by path - and nothing else. An exemption
list that grows past one entry means the checker is wrong or the code is.

Usage: python tools/lint_human.py   (exits non-zero on failure)
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXEMPT = {REPO / "docs" / "09_HUMAN_CODE.md"}

_VOCAB_DOC = (REPO / "docs" / "09_HUMAN_CODE.md").read_text(encoding="utf-8")
_doc_lines = _VOCAB_DOC.splitlines()
_heading = _doc_lines.index("## Vocabulary to avoid")
_start = next(
    i for i in range(_heading + 1, len(_doc_lines))
    if _doc_lines[i].startswith("*") and "," in _doc_lines[i]
)
_block = ""
for line in _doc_lines[_start:]:
    _block += line.strip("* \n")
    if line.rstrip().endswith("*"):
        break
FORBIDDEN_WORDS = [word.strip() for word in _block.split(",")]
_WORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in FORBIDDEN_WORDS) + r")\b",
    re.IGNORECASE,
)
_EMOJI_RE = re.compile(
    "[\U0001f000-\U0001faff\U00002600-\U000027bf\U0001f900-\U0001f9ff]"
)
_TRAILER_RE = re.compile(
    r"^\s*(Co-Authored-By|Signed-off-by|Generated with|Authored by)\b",
    re.IGNORECASE | re.MULTILINE,
)
_CHANGELOG_RE = re.compile(
    r"^\s*#+\s*changelog", re.IGNORECASE | re.MULTILINE
)
_BANNER_RE = re.compile(r"^\s*#\s*[-=]{20,}\s*$", re.MULTILINE)

FAILURES: list[str] = []


def scan(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    label = str(path.relative_to(REPO))
    for match in _WORD_RE.finditer(text):
        FAILURES.append(f"{label}:{line_no(text, match.start())}: forbidden word {match.group(0)!r}")
    for match in _EMOJI_RE.finditer(text):
        FAILURES.append(f"{label}:{line_no(text, match.start())}: emoji {match.group(0)!r}")
    for match in _TRAILER_RE.finditer(text):
        FAILURES.append(f"{label}:{line_no(text, match.start())}: attribution trailer {match.group(1)}")
    for match in _CHANGELOG_RE.finditer(text):
        FAILURES.append(f"{label}:{line_no(text, match.start())}: changelog header")
    for match in _BANNER_RE.finditer(text):
        FAILURES.append(f"{label}:{line_no(text, match.start())}: decorative banner comment")


def line_no(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def main():
    print("tools/lint_human.py - style checks (docs/09)")
    checked_suffixes = {".md", ".py", ".txt", ".yaml", ".yml"}
    files = [
        path
        for path in REPO.rglob("*")
        if path.is_file()
        and path.suffix in checked_suffixes
        and path not in EXEMPT
        and "__pycache__" not in path.parts
        and ".git" not in path.parts
    ]
    for path in sorted(files):
        scan(path)
    if FAILURES:
        for failure in FAILURES:
            print(f"FAIL  {failure}")
        print(f"\n{len(FAILURES)} style violation(s)")
        sys.exit(1)
    print("all style checks pass")
    print(f"files scanned: {len(files)}")


if __name__ == "__main__":
    main()
