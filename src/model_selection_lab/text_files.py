"""Reading text and record files, some of them private.

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


def read_records(path, build) -> list:
    """Read a .jsonl file (one JSON object per line) and turn each line into a record with
    build(line). A line that cannot be read (cut off by a crash, or from another version of
    the tool) is skipped with a printed warning, so one bad line cannot break everything."""
    records = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(build(line))
        except (ValueError, TypeError) as problem:  # bad JSON, or fields that do not match
            print(f"Skipped an unreadable line {number} in {path}: {problem}")
    return records
