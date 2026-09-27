"""Consistent output-path handling for runnable examples."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "output"


def output_path(path: Path | None, default_name: str) -> Path:
    """Return an example output path rooted in the repository ``output/`` folder.

    Absolute paths remain explicit user overrides.  Relative paths are placed
    below ``output/``; paths already beginning with ``output/`` are interpreted
    relative to the repository root to avoid producing ``output/output/...``.
    """
    if path is None:
        return OUTPUT_ROOT / default_name
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "output":
        return PROJECT_ROOT / path
    return OUTPUT_ROOT / path
