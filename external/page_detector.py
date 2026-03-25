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
    "file": re.compile(r"^FILE=(.+)$", re.MULTILINE),
    "name": re.compile(r"^NAME=(.+)$", re.MULTILINE),
    "page": re.compile(r"^PAGE=(\d+)/(\d+)$", re.MULTILINE),
    "lines": re.compile(r"^LINES=(\d+)-(\d+)$", re.MULTILINE),
}


def parse_header(text: str) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, pattern in HEADER_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            continue
        if key in {"page", "lines"}:
            result[key] = tuple(int(item) for item in match.groups())
        else:
            result[key] = match.group(1).strip()
    return result


def page_identity(header: Dict[str, object]) -> Optional[str]:
    file_path = header.get("file")
    lines = header.get("lines")
    page_info = header.get("page")
    if not file_path or not lines:
        return None
    if page_info:
        return "%s::page-%s" % (file_path, page_info[0])
    return "%s::%s-%s" % (file_path, lines[0], lines[1])


def is_last_page(header: Dict[str, object]) -> bool:
    page_info = header.get("page")
    if not page_info:
        return False
    return int(page_info[0]) == int(page_info[1])


def is_new_page(previous_identity: Optional[str], current_header: Dict[str, object]) -> bool:
    current_identity = page_identity(current_header)
    if not current_identity:
        return False
    return current_identity != previous_identity
