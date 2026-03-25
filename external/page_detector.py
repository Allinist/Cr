#!/usr/bin/env python3
"""
Detect whether a screenshot represents a new page.

This version focuses on metadata parsing from OCR text rather than pixel-level
 scene detection so it stays simple and debuggable.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple


HEADER_PATTERNS = {
    "file": re.compile(r"^FILE\s*[:=-]\s*(.+)$", re.MULTILINE | re.IGNORECASE),
    "name": re.compile(r"^NAME\s*[:=-]\s*(.+)$", re.MULTILINE | re.IGNORECASE),
    "page": re.compile(r"^PAGE\s*[:=-]\s*(\d+)\s*/\s*(\d+)$", re.MULTILINE | re.IGNORECASE),
    "lines": re.compile(r"^LINES\s*[:=-]\s*(\d+)\s*-\s*(\d+)$", re.MULTILINE | re.IGNORECASE),
}

PAGE_BEGIN_PATTERN = re.compile(
    r"\[PAGE-BEGIN\]\s+file=(.+?)\s+page=(\d+)/(\d+)\s+lines=(\d+)-(\d+)"
)
PAGE_END_PATTERN = re.compile(
    r"\[PAGE-END\]\s+file=(.+?)\s+page=(\d+)/(\d+)\s+lines=(\d+)-(\d+)"
)
FUZZY_FILE_PATTERN = re.compile(r"(?:FILE|F1LE|FILE|RILE|FIIE|fiie)\s*[-=:]?\s*([^\n]+)", re.IGNORECASE)
FUZZY_PAGE_PATTERN = re.compile(r"PAGE\s*[-=:]?\s*([0-9Il|])\s*([/7])\s*([0-9Il|])", re.IGNORECASE)
FUZZY_LINES_PATTERN = re.compile(r"(?:LINES|1INES)\s*[-=:]?\s*(\d+)\s*[-/]\s*(\d+)", re.IGNORECASE)


def normalize_digit_token(token: str) -> int:
    normalized = (
        token.replace("I", "1")
        .replace("l", "1")
        .replace("|", "1")
        .replace("O", "0")
        .replace("o", "0")
    )
    return int(normalized)


def cleanup_file_value(value: str) -> str:
    cleaned = value.strip()
    cleaned = cleaned.replace("\\", "/")
    cleaned = cleaned.replace(" ", "")
    cleaned = cleaned.replace(",", ".")
    cleaned = cleaned.replace(";", "")
    cleaned = cleaned.replace(":", "/")
    cleaned = re.sub(r"^[^A-Za-z0-9./_-]+", "", cleaned)
    cleaned = re.sub(r"^(?:file|fiie|f1le|ri1e|rile)", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("java7com", "java/com")
    cleaned = cleaned.replace("src/main/fava", "src/main/java")
    cleaned = cleaned.replace("src/main/iava", "src/main/java")
    cleaned = cleaned.replace("sre/main/java", "src/main/java")
    cleaned = cleaned.replace("sre/main/iava", "src/main/java")
    cleaned = cleaned.replace("sic/main/java", "src/main/java")
    cleaned = cleaned.replace("tom/spo", "com/spc")
    cleaned = cleaned.replace("tom/spc", "com/spc")
    cleaned = cleaned.replace("/con/", "/com/")
    cleaned = cleaned.replace("/spo/", "/spc/")
    cleaned = cleaned.replace("/sps-", "/spc-")
    cleaned = cleaned.replace("spe-iodules", "spc-modules")
    cleaned = cleaned.replace("spe-moduies", "spc-modules")
    cleaned = cleaned.replace("sps-modules", "spc-modules")
    cleaned = cleaned.replace("spe-modules", "spc-modules")
    cleaned = cleaned.replace("m6dules", "modules")
    cleaned = cleaned.replace("ivstcanv", "ivstconv")
    cleaned = cleaned.replace("iystconv", "ivstconv")
    cleaned = cleaned.replace("controiler", "controller")
    cleaned = cleaned.replace("contro1ler", "controller")
    cleaned = cleaned.replace("controlier", "controller")
    cleaned = cleaned.replace("Contro1ler", "Controller")
    cleaned = cleaned.replace("Fiie", "File")
    cleaned = cleaned.replace("fiie", "file")
    cleaned = cleaned.replace("Fileinfo", "FileInfo")
    cleaned = cleaned.replace("fileinfo", "FileInfo")
    cleaned = re.sub(r"(page[-=:]?\d+[\/7]\d+.*)$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(1ines[-=:]?\d+[-/]\d+.*)$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(lines[-=:]?\d+[-/]\d+.*)$", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("-java", ".java")
    cleaned = cleaned.replace(",java", ".java")
    cleaned = re.sub(r"/+", "/", cleaned)
    cleaned = cleaned.rstrip("./-")
    return cleaned


def score_file_candidate(value: str) -> int:
    score = 0
    lower_value = value.lower()
    if ".java" in lower_value:
        score += 5
    if "/src/main/java/" in lower_value:
        score += 5
    if "/controller/" in lower_value:
        score += 3
    if "spc-modules" in lower_value:
        score += 2
    score += min(lower_value.count("/"), 8)
    score -= len(re.findall(r"[^a-z0-9./_-]", lower_value))
    return score


def normalize_page_tuple(groups: Tuple[str, str]) -> Tuple[int, int]:
    return (normalize_digit_token(groups[0]), normalize_digit_token(groups[1]))


def pick_best_file_candidate(candidates: List[str]) -> Optional[str]:
    normalized = []
    for candidate in candidates:
        cleaned = cleanup_file_value(candidate)
        if not cleaned:
            continue
        normalized.append(cleaned)
    if not normalized:
        return None
    normalized.sort(key=lambda item: (score_file_candidate(item), len(item)), reverse=True)
    return normalized[0]


def parse_header(text: str) -> Dict[str, object]:
    result: Dict[str, object] = {}
    file_candidates: List[str] = []

    marker_match = PAGE_BEGIN_PATTERN.search(text)
    if marker_match:
        file_candidates.append(marker_match.group(1))
        result["page"] = (int(marker_match.group(2)), int(marker_match.group(3)))
        result["lines"] = (int(marker_match.group(4)), int(marker_match.group(5)))

    footer_match = PAGE_END_PATTERN.search(text)
    if footer_match:
        file_candidates.append(footer_match.group(1))
        result.setdefault("page", (int(footer_match.group(2)), int(footer_match.group(3))))
        result.setdefault("lines", (int(footer_match.group(4)), int(footer_match.group(5))))

    exact_file = HEADER_PATTERNS["file"].search(text)
    if exact_file:
        file_candidates.append(exact_file.group(1))
    exact_name = HEADER_PATTERNS["name"].search(text)
    if exact_name:
        result["name"] = exact_name.group(1).strip()
    exact_page = HEADER_PATTERNS["page"].search(text)
    if exact_page:
        result["page"] = (int(exact_page.group(1)), int(exact_page.group(2)))
    exact_lines = HEADER_PATTERNS["lines"].search(text)
    if exact_lines:
        result["lines"] = (int(exact_lines.group(1)), int(exact_lines.group(2)))

    fuzzy_file = FUZZY_FILE_PATTERN.search(text)
    if fuzzy_file:
        file_candidates.append(fuzzy_file.group(1))
    fuzzy_page = FUZZY_PAGE_PATTERN.search(text)
    if "page" not in result and fuzzy_page:
        result["page"] = normalize_page_tuple((fuzzy_page.group(1), fuzzy_page.group(3)))
    fuzzy_lines = FUZZY_LINES_PATTERN.search(text)
    if "lines" not in result and fuzzy_lines:
        result["lines"] = (int(fuzzy_lines.group(1)), int(fuzzy_lines.group(2)))

    best_file = pick_best_file_candidate(file_candidates)
    if best_file:
        result["file"] = best_file
    if "name" not in result and "file" in result:
        result["name"] = os.path.basename(str(result["file"]))
    return result


def has_page_markers(text: str) -> bool:
    if PAGE_BEGIN_PATTERN.search(text) is not None and PAGE_END_PATTERN.search(text) is not None:
        return True
    upper_text = text.upper()
    return "PAGE-BEGIN" in upper_text and "PAGE-END" in upper_text


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
