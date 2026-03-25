#!/usr/bin/env python3
"""
Rebuild file content from OCR page chunks.

This is intentionally conservative for v1: it merges chunk text into a
 structured intermediate format and leaves language-specific repair hooks for
 the next step.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

from page_detector import extract_body_region, parse_header


LINE_PATTERN = re.compile(r"^\s*(\d+)\s{2}(.*)$")
CONTINUATION_PATTERN = re.compile(r"^\s{5}\s{2}(.*)$")
DIGIT_ONLY_PATTERN = re.compile(r"^\d{1,4}$")
LEADING_LINE_PATTERN = re.compile(r"^\s*(\d{1,4})([A-Za-z_@/*$].*)$")
COMMENT_MARKER_PATTERN = re.compile(r"^(?:/\*\*?|//+|\*+/|\*|[*]\s*.+)$")
CHINESE_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fff]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild code files from OCR JSON.")
    parser.add_argument("--input", required=True, help="OCR JSON path.")
    parser.add_argument("--output", required=True, help="Rebuilt JSON path.")
    return parser.parse_args()


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def load_payload(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_entries(path: str) -> List[Dict[str, object]]:
    payload = load_payload(path)
    if payload.get("regions", {}).get("body", {}).get("entries"):
        return payload["regions"]["body"]["entries"]
    return payload.get("entries", [])


def polygon_center_y(entry: Dict[str, object]) -> float:
    polygon = entry.get("polygon") or []
    if not polygon:
        return 0.0
    return sum(float(point[1]) for point in polygon) / len(polygon)


def group_entries_by_line(entries: List[Dict[str, object]], threshold: float = 14.0) -> List[List[Dict[str, object]]]:
    sorted_entries = sorted(entries, key=lambda item: polygon_center_y(item))
    groups: List[List[Dict[str, object]]] = []
    for entry in sorted_entries:
        if not groups:
            groups.append([entry])
            continue
        current_group = groups[-1]
        avg_y = sum(polygon_center_y(item) for item in current_group) / len(current_group)
        if abs(polygon_center_y(entry) - avg_y) <= threshold:
            current_group.append(entry)
        else:
            groups.append([entry])
    return groups


def group_text(group: List[Dict[str, object]]) -> str:
    sorted_group = sorted(
        group,
        key=lambda item: min(float(point[0]) for point in (item.get("polygon") or [[0, 0]])),
    )
    return " ".join(str(item.get("text", "")).strip() for item in sorted_group if str(item.get("text", "")).strip()).strip()


def polygon_min_x(entry: Dict[str, object]) -> float:
    polygon = entry.get("polygon") or []
    if not polygon:
        return 0.0
    return min(float(point[0]) for point in polygon)


def looks_like_noise(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return True
    if COMMENT_MARKER_PATTERN.match(text.strip()):
        return False
    if len(compact) <= 2:
        return True
    if compact.startswith("[PAGE-") or compact.startswith("FILE=") or compact.startswith("PAGE=") or compact.startswith("LINES="):
        return True
    alpha_count = sum(1 for ch in compact if ch.isalpha())
    digit_count = sum(1 for ch in compact if ch.isdigit())
    punctuation_count = len(compact) - alpha_count - digit_count
    if alpha_count == 0 and digit_count == 0:
        return True
    if alpha_count == 0 and punctuation_count > 0:
        return True
    return False


def clean_code_text(text: str) -> str:
    raw = re.sub(r"\s+", " ", text).strip()
    if raw in {"/**", "/*", "*/", "*"}:
        return raw
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = cleaned.replace("mport", "import")
    cleaned = cleaned.replace("inport", "import")
    cleaned = cleaned.replace("fava", "java")
    cleaned = cleaned.replace("iava", "java")
    cleaned = cleaned.replace("Fiie", "File")
    cleaned = cleaned.replace("Contro1ler", "Controller")
    cleaned = cleaned.replace("contro1ler", "controller")
    cleaned = cleaned.replace("controlier", "controller")
    cleaned = cleaned.replace("orgsspringframework", "org.springframework")
    cleaned = cleaned.replace("org-springframework", "org.springframework")
    cleaned = cleaned.replace("veb", "web")
    cleaned = cleaned.replace("annotations.", "annotations.")
    cleaned = cleaned.replace("Restcontroller", "RestController")
    cleaned = cleaned.replace("pibiic", "public")
    cleaned = cleaned.replace("ciass", "class")
    cleaned = re.sub(r"^\+\s*", "* ", cleaned)
    cleaned = re.sub(r"^(?:[1il]+\s*)?import\b", "import", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:[1il]+\s*)?@RestController\b", "@RestController", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^([*])(?=[^\s/])", r"\1 ", cleaned)
    cleaned = cleaned.strip(" .,:;")
    return cleaned


def expected_line_range(payload: Dict[str, object]) -> Optional[Tuple[int, int]]:
    regions = payload.get("regions", {})
    header_text = str(regions.get("header", {}).get("text", "")).strip()
    footer_text = str(regions.get("footer", {}).get("text", "")).strip()
    body_text = str(regions.get("body", {}).get("text", "")).strip()
    header = parse_header("\n".join(part for part in [header_text, body_text[:200], footer_text] if part))
    lines = header.get("lines")
    if not lines:
        return None
    return (int(lines[0]), int(lines[1]))


def upsert_line(lines_by_no: Dict[int, str], line_no: int, text: str) -> None:
    if line_no <= 0:
        return
    cleaned = clean_code_text(text)
    if looks_like_noise(cleaned):
        return
    previous = lines_by_no.get(line_no)
    if previous is None or len(cleaned) > len(previous):
        lines_by_no[line_no] = cleaned


def infer_group_line_no(group: List[Dict[str, object]]) -> Tuple[Optional[int], str, float]:
    sorted_group = sorted(group, key=polygon_min_x)
    parts = [str(item.get("text", "")).strip() for item in sorted_group if str(item.get("text", "")).strip()]
    if not parts:
        return (None, "", 0.0)
    line_no = None
    code_parts = parts
    first_text = parts[0]
    if DIGIT_ONLY_PATTERN.match(first_text):
        line_no = int(first_text)
        code_parts = parts[1:]
    else:
        leading_match = LEADING_LINE_PATTERN.match(first_text)
        if leading_match and polygon_min_x(sorted_group[0]) <= 140.0:
            line_no = int(leading_match.group(1))
            code_parts = [leading_match.group(2).strip()] + parts[1:]
    avg_y = sum(polygon_center_y(item) for item in group) / len(group)
    candidate_text = " ".join(part for part in code_parts if part).strip()
    return (line_no, candidate_text, avg_y)


def fill_missing_line_numbers(
    ordered_groups: List[Dict[str, object]], line_start: int, line_end: int
) -> List[Tuple[int, str]]:
    if not ordered_groups:
        return []

    assigned: List[Tuple[int, str]] = []
    indexed = list(enumerate(ordered_groups))
    explicit = [(idx, int(item["line_no"])) for idx, item in indexed if item.get("line_no") is not None]

    if not explicit:
        current_line = line_start
        for _, item in indexed:
            if current_line > line_end:
                break
            assigned.append((current_line, str(item["text"])))
            current_line += 1
        return assigned

    first_idx, first_line = explicit[0]
    current_line = max(line_start, first_line - first_idx)
    for idx in range(0, first_idx):
        if current_line > line_end:
            break
        assigned.append((current_line, str(ordered_groups[idx]["text"])))
        current_line += 1

    for position, (anchor_idx, anchor_line) in enumerate(explicit):
        assigned.append((anchor_line, str(ordered_groups[anchor_idx]["text"])))
        next_idx = explicit[position + 1][0] if position + 1 < len(explicit) else None
        next_line = explicit[position + 1][1] if position + 1 < len(explicit) else None
        if next_idx is None or next_line is None:
            current_line = anchor_line + 1
            for idx in range(anchor_idx + 1, len(ordered_groups)):
                if current_line > line_end:
                    break
                assigned.append((current_line, str(ordered_groups[idx]["text"])))
                current_line += 1
            break

        gap_indices = list(range(anchor_idx + 1, next_idx))
        if not gap_indices:
            continue
        gap_line_capacity = max(0, next_line - anchor_line - 1)
        if gap_line_capacity == 0:
            continue
        if len(gap_indices) <= gap_line_capacity:
            current_line = anchor_line + 1
            for idx in gap_indices:
                assigned.append((current_line, str(ordered_groups[idx]["text"])))
                current_line += 1
        else:
            step = max(1.0, float(gap_line_capacity) / float(len(gap_indices)))
            for offset, idx in enumerate(gap_indices, start=1):
                inferred = anchor_line + min(gap_line_capacity, max(1, int(round(offset * step))))
                assigned.append((inferred, str(ordered_groups[idx]["text"])))

    assigned.sort(key=lambda item: item[0])
    deduped: Dict[int, str] = {}
    for line_no, text in assigned:
        upsert_line(deduped, line_no, text)
    return sorted(deduped.items())


def looks_like_doc_comment_content(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if lowered.startswith("*") or lowered.startswith("/**") or lowered.startswith("/*") or lowered.startswith("*/"):
        return True
    if "@author" in lowered or "@date" in lowered:
        return True
    if CHINESE_CHAR_PATTERN.search(stripped) is not None:
        return True
    return False


def normalize_doc_comment_blocks(lines_by_no: Dict[int, str], line_start: int, line_end: int) -> Dict[int, str]:
    normalized = dict(lines_by_no)
    candidate_lines = sorted(
        line_no for line_no, text in normalized.items() if line_start <= line_no <= line_end and looks_like_doc_comment_content(text)
    )
    if not candidate_lines:
        return normalized

    blocks: List[List[int]] = []
    for line_no in candidate_lines:
        if not blocks or line_no - blocks[-1][-1] > 2:
            blocks.append([line_no])
        else:
            blocks[-1].append(line_no)

    for block in blocks:
        block_start = block[0]
        block_end = block[-1]
        comment_start = max(line_start, block_start - 1)
        comment_end = min(line_end, block_end + 1)

        if comment_start not in normalized:
            normalized[comment_start] = "/**"
        elif normalized[comment_start].strip() not in {"/**", "/*"} and comment_start < block_start:
            normalized[comment_start] = "/**"

        for line_no in block:
            text = normalized.get(line_no, "").strip()
            if text in {"/**", "/*", "*/"}:
                continue
            if text.startswith("*"):
                continue
            normalized[line_no] = "* " + text.lstrip("+").strip()

        if comment_end not in normalized:
            normalized[comment_end] = "*/"
        elif normalized[comment_end].strip() not in {"*/"} and comment_end > block_end:
            normalized[comment_end] = "*/"

    cleaned: Dict[int, str] = {}
    for line_no, text in normalized.items():
        upsert_line(cleaned, line_no, text)
    return cleaned


def parse_structured_body(payload: Dict[str, object]) -> Optional[List[Tuple[int, str]]]:
    regions = payload.get("regions", {})
    body_entries = list(regions.get("body", {}).get("entries", []))
    line_entries = list(regions.get("body_line_numbers", {}).get("entries", []))
    code_entries = list(regions.get("body_code", {}).get("entries", []))
    if not body_entries and (not line_entries or not code_entries):
        return None

    number_candidates = []
    source_line_entries = line_entries or body_entries
    for entry in source_line_entries:
        text = str(entry.get("text", "")).strip()
        if DIGIT_ONLY_PATTERN.match(text):
            number_candidates.append(
                {
                    "line_no": int(text),
                    "y": polygon_center_y(entry),
                    "x": polygon_min_x(entry),
                }
            )
            continue
        leading_match = LEADING_LINE_PATTERN.match(text)
        if not leading_match:
            continue
        number_candidates.append(
            {
                "line_no": int(leading_match.group(1)),
                "y": polygon_center_y(entry),
                "x": polygon_min_x(entry),
            }
        )
    if not number_candidates:
        return None

    lines_by_no: Dict[int, str] = {}
    line_range = expected_line_range(payload)
    ordered_groups: List[Dict[str, object]] = []

    if body_entries:
        body_groups = group_entries_by_line(body_entries)
        for group in body_groups:
            line_no, candidate_text, avg_y = infer_group_line_no(group)
            if candidate_text:
                ordered_groups.append({"line_no": line_no, "text": candidate_text, "y": avg_y})
            if line_no is None:
                continue
            upsert_line(lines_by_no, line_no, candidate_text)

    code_groups = group_entries_by_line(code_entries or body_entries)
    for group in code_groups:
        text = clean_code_text(group_text(group))
        if looks_like_noise(text):
            continue
        avg_y = sum(polygon_center_y(item) for item in group) / len(group)
        best_index = None
        best_distance = None
        for index, candidate in enumerate(number_candidates):
            distance = abs(candidate["y"] - avg_y)
            if best_distance is None or distance < best_distance:
                best_index = index
                best_distance = distance
        if best_index is None or best_distance is None or best_distance > 24.0:
            continue
        candidate = number_candidates[best_index]
        upsert_line(lines_by_no, int(candidate["line_no"]), text)

    if line_range is not None and ordered_groups:
        inferred_lines = fill_missing_line_numbers(
            sorted(ordered_groups, key=lambda item: float(item["y"])),
            int(line_range[0]),
            int(line_range[1]),
        )
        for line_no, text in inferred_lines:
            upsert_line(lines_by_no, line_no, text)
        lines_by_no = normalize_doc_comment_blocks(lines_by_no, int(line_range[0]), int(line_range[1]))

    if not lines_by_no:
        return None
    return sorted((line_no, text) for line_no, text in lines_by_no.items())


def parse_body_lines(entries: List[Dict[str, object]]) -> List[Tuple[int, str]]:
    logical_lines: List[List[object]] = []
    for entry in entries:
        raw_text = extract_body_region(str(entry.get("text", "")))
        for text in raw_text.splitlines():
            stripped = text.strip()
            if not stripped:
                continue
            if stripped.startswith("[PAGE-BEGIN]") or stripped.startswith("[PAGE-END]"):
                continue
            if stripped.startswith("FILE=") or stripped.startswith("NAME=") or stripped.startswith("PAGE=") or stripped.startswith("LINES="):
                continue
            if set(stripped) == {"="}:
                continue
            match = LINE_PATTERN.match(text)
            if match:
                logical_lines.append([int(match.group(1)), match.group(2)])
                continue
            continuation = CONTINUATION_PATTERN.match(text)
            if continuation and logical_lines:
                logical_lines[-1][1] = str(logical_lines[-1][1]) + continuation.group(1)
    return [(int(item[0]), str(item[1])) for item in logical_lines]


def main() -> int:
    args = parse_args()
    payload = load_payload(args.input)
    body_lines = parse_structured_body(payload)
    if body_lines is None:
        entries = load_entries(args.input)
        body_lines = parse_body_lines(entries)
    line_range = expected_line_range(payload)
    if line_range is not None:
        start_line, end_line = line_range
        body_lines = [
            (line_no, text)
            for line_no, text in body_lines
            if start_line <= int(line_no) <= end_line
        ]
    payload = {
        "source": args.input,
        "line_count": len(body_lines),
        "lines": [{"line_no": line_no, "text": text} for line_no, text in body_lines],
    }
    ensure_parent(args.output)
    with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
