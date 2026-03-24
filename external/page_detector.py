#!/usr/bin/env python3
"""
Detect whether a screenshot represents a new page.

This version focuses on metadata parsing from OCR text rather than pixel-level
 scene detection so it stays simple and debuggable.
"""

from __future__ import annotations

import re
from typing import Dict, Optional


HEADER_PATTERNS = {
    "repo": re.compile(r"^REPO=(.+)$", re.MULTILINE),
    "branch": re.compile(r"^BRANCH=(.+)$", re.MULTILINE),
    "commit": re.compile(r"^COMMIT=(.+)$", re.MULTILINE),
    "file": re.compile(r"^FILE=(.+)$", re.MULTILINE),
    "language": re.compile(r"^LANG=(.+)$", re.MULTILINE),
    "page": re.compile(r"^PAGE=(\d+)/(\d+)$", re.MULTILINE),
    "lines": re.compile(r"^LINES=(\d+)-(\d+)$", re.MULTILINE),
    "slice": re.compile(r"^SLICE=(\d+)/(\d+)$", re.MULTILINE),
    "cols": re.compile(r"^COLS=(\d+)-(\d+)$", re.MULTILINE),
    "chunk": re.compile(r"^CHUNK=([a-fA-F0-9]+)$", re.MULTILINE),
}


def parse_header(text: str) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, pattern in HEADER_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            continue
        if key in {"page", "lines", "slice", "cols"}:
            result[key] = tuple(int(item) for item in match.groups())
        else:
            result[key] = match.group(1).strip()
    return result


def page_identity(header: Dict[str, object]) -> Optional[str]:
    file_path = header.get("file")
    chunk = header.get("chunk")
    if not file_path or not chunk:
        return None
    return "%s::%s" % (file_path, chunk)


def is_new_page(previous_identity: Optional[str], current_header: Dict[str, object]) -> bool:
    current_identity = page_identity(current_header)
    if not current_identity:
        return False
    return current_identity != previous_identity
