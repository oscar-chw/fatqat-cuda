"""Refuse benchmark output or docs that would publish machine details.

Results published from this fork describe the run configuration ("one NVIDIA
GPU", "compiled Numba, 8 threads") but never the machine: no hardware model,
core or memory size, host name, IP address, filesystem path or device ID.
The benchmark scripts never record those fields; this check is the backstop
that runs on every file before it is committed.

The patterns below are generic. Site-specific names (host names, user names)
cannot live in a public file, so they are read from an optional private
pattern file passed with ``--private``.

Exit status: 0 clean, 1 at least one finding, 2 a named file is missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

# Each entry: (label, pattern). Patterns are case-insensitive.
GENERIC_PATTERNS: tuple[tuple[str, str], ...] = (
    ("gpu model", r"\b(?:rtx|gtx|geforce|quadro|tesla|titan)\b"),
    ("gpu model", r"\b(?:[ahlbv]100|h200|gh200|b200|gb200|l40s?|a[1-9]0{2,3})\b"),
    ("gpu model", r"\bnvidia\s+(?!gpu\b)[a-z]*\d"),
    ("cpu model", r"\b(?:xeon|epyc|threadripper|ryzen|core\s+i[3579])\b"),
    ("cpu model", r"\bapple\s+m\d\b|\bm[1-9]\s+(?:pro|max|ultra)\b"),
    ("core count", r"\b\d+\s*[- ]?(?:physical\s+|logical\s+)?cores?\b"),
    (
        "memory size",
        r"\b\d+(?:\.\d+)?\s*(?:gb|gib|tb|tib)\s+(?:of\s+)?(?:v?ram|memory)\b",
    ),
    ("device count", r"\b\d+\s*[x×]\s*(?:gpus?|cards?)\b"),
    ("ip address", r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ("home path", r"(?:/home/|/Users/|[A-Za-z]:\\Users\\)[^\s\"'/\\]+"),
    ("device id", r"\bGPU-[0-9a-f]{8}-"),
    ("pci bus id", r"\b[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]\b"),
    ("driver version", r"\bdriver(?:\s+version)?[\s:=]+\d{3}\.\d+"),
    ("hostname field", r"[\"']?(?:host(?:name)?|node|machine)[\"']?\s*[:=]\s*[\"']?\w"),
)


def load_private(path: Path) -> list[tuple[str, str]]:
    """Read one regex per line; blank and '#' lines are ignored."""
    patterns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.append(("private pattern", line))
    return patterns


def scan(text: str, patterns) -> list[tuple[int, str, str]]:
    """Return (line number, label, matched text) for every finding."""
    compiled = [(label, re.compile(p, re.IGNORECASE)) for label, p in patterns]
    findings = []
    for number, line in enumerate(text.splitlines(), start=1):
        for label, regex in compiled:
            for match in regex.finditer(line):
                findings.append((number, label, match.group(0)))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument(
        "--private",
        type=Path,
        help="file of extra site-specific regexes, kept outside the repository",
    )
    args = parser.parse_args(argv)

    patterns = list(GENERIC_PATTERNS)
    if args.private is not None:
        # A named private file that is missing must fail: silently scanning
        # with fewer patterns would read as a clean result.
        if not args.private.is_file():
            print(f"scrub_check: private pattern file not found: {args.private}")
            return 2
        patterns += load_private(args.private)

    status = 0
    for path in args.files:
        if not path.is_file():
            print(f"scrub_check: missing file: {path}")
            return 2
        findings = scan(path.read_text(encoding="utf-8"), patterns)
        for number, label, matched in findings:
            # Print only the label for private patterns so the check itself
            # never echoes a private name into a log.
            shown = "<private>" if label == "private pattern" else repr(matched)
            print(f"{path}:{number}: {label}: {shown}")
        if findings:
            status = 1
    if status == 0:
        print(
            f"scrub_check: clean ({len(args.files)} file(s), {len(patterns)} patterns)"
        )
    return status


if __name__ == "__main__":
    sys.exit(main())
