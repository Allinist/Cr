#!/usr/bin/env python3
"""
Detect whether a screenshot represents a new page.

This version focuses on metadata parsing from OCR text rather than pixel-level
 scene detection so it stays simple and debuggable.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional


HEADER_PATTERNS = {
    "file": re.compile(r"^FILE=(.+)$", re.MULTILINE),
    "name": re.compile(r"^NAME=(.+)$", re.MULTILINE),
    "page": re.compile(r"^PAGE=(\d+)/(\d+)$", re.MULTILINE),
    "lines": re.compile(r"^LINES=(\d+)-(\d+)$", re.MULTILINE),
}

PAGE_BEGIN_PATTERN = re.compile(
    r"\[PAGE-BEGIN\]\s+file=(.+?)\s+page=(\d+)/(\d+)\s+lines=(\d+)-(\d+)"
)
PAGE_END_PATTERN = re.compile(
    r"\[PAGE-END\]\s+file=(.+?)\s+page=(\d+)/(\d+)\s+lines=(\d+)-(\d+)"
)


def parse_header(text: str) -> Dict[str, object]:
    result: Dict[str, object] = {}
    marker_match = PAGE_BEGIN_PATTERN.search(text)
    if marker_match:
        result["file"] = marker_match.group(1).strip()
        result["name"] = result["file"].rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        result["page"] = (int(marker_match.group(2)), int(marker_match.group(3)))
        result["lines"] = (int(marker_match.group(4)), int(marker_match.group(5)))
    for key, pattern in HEADER_PATTERNS.items():
        if key in result:
            continue
        match = pattern.search(text)
        if not match:
            continue
        if key in {"page", "lines"}:
            result[key] = tuple(int(item) for item in match.groups())
        else:
            result[key] = match.group(1).strip()
    if "name" not in result and "file" in result:
        result["name"] = str(result["file"]).rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    return result


def has_page_markers(text: str) -> bool:
    return PAGE_BEGIN_PATTERN.search(text) is not None and PAGE_END_PATTERN.search(text) is not None


def extract_body_region(text: str) -> str:
    begin_match = PAGE_BEGIN_PATTERN.search(text)
    end_match = PAGE_END_PATTERN.search(text)
    start_index = begin_match.end() if begin_match else 0
    end_index = end_match.start() if end_match else len(text)
    return text[start_index:end_index].strip("\n")


def split_lines(text: str) -> List[str]:
    return [line.rstrip("\r") for line in text.splitlines()]


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
