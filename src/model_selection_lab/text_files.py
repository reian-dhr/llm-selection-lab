"""Reading text files that may be private.

Prompts and judge rules are personal: each user can keep their own tuned versions in
data/, which is not part of the repository. Generic versions ship under examples/. Code
that needs such a file reads the personal one if it exists, otherwise the example.
"""

from pathlib import Path


def read_text(path: str, fallback: str = "") -> str:
    """Read a text file, or the fallback file if the first one does not exist."""
    if Path(path).exists():
        return Path(path).read_text(encoding="utf-8")
    return Path(fallback).read_text(encoding="utf-8") if fallback else ""
