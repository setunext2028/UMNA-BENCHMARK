"""Emits the one comparison table (S5)."""

from __future__ import annotations

from bench.harness import run_all, ArmResult


def emit_table(results: dict[str, dict[str, ArmResult]]) -> None:
    """Emit the comparison table in a format suitable for the paper."""
    pass


def main() -> None:
    run_all()


if __name__ == "__main__":
    main()