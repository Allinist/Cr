#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

from roi_code_rebuilder import replace_cjk_punctuation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fused code files from OCR JSON outputs.")
    parser.add_argument("--ocr-root", required=True, help="Root directory containing *.ocr.json files.")
    parser.add_argument("--output-root", required=True, help="Root directory for fused code files.")
    return parser.parse_args()


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)


def load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_ocr_files(root: str) -> List[str]:
    matches: List[str] = []
    for current_root, _, files in os.walk(root):
        for name in files:
            if name.endswith(".ocr.json"):
                matches.append(os.path.join(current_root, name))
    matches.sort()
    return matches


def normalize_path_output(ocr_root: str, output_root: str, ocr_path: str) -> str:
    rel_path = os.path.relpath(ocr_path, ocr_root)
    if rel_path.endswith(".ocr.json"):
        rel_path = rel_path[:-9]
    return os.path.join(output_root, rel_path)


def write_text(path: str, text: str) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text.rstrip() + "\n")


def classify_line(text: str) -> str:
    stripped = str(text).strip()
    if not stripped:
        return "blank"
    if stripped.startswith("package ") or stripped.startswith("package"):
        return "package"
    if stripped.startswith("import ") or stripped.startswith("import"):
        return "import"
    if stripped.startswith("@"):
        return "annotation"
    if stripped.startswith("*") or stripped in {"/**", "/*", "*/"}:
        return "comment"
    return "code"


def normalize_light(text: str) -> str:
    text = replace_cjk_punctuation(str(text))
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^package(?=[A-Za-z_])", "package ", text)
    text = re.sub(r"^import(?=[A-Za-z_])", "import ", text)
    text = re.sub(r"^public(?=class\b|interface\b|enum\b)", "public ", text)
    text = re.sub(r"^publicstatic", "public static", text)
    text = re.sub(r"^publicfinal", "public final", text)
    text = re.sub(r"^privatefinal", "private final", text)
    text = re.sub(r"^protectedfinal", "protected final", text)
    text = re.sub(r"^publicinterface", "public interface", text)
    text = re.sub(r"^publicclass", "public class", text)
    text = re.sub(r"^publicenum", "public enum", text)
    text = re.sub(r"^import\s+", "import ", text)
    text = re.sub(r"^package\s+", "package ", text)
    return text.strip()


def normalize_import_or_package(text: str) -> str:
    text = normalize_light(text)
    if text.startswith("import ") and not text.endswith(";"):
        text += ";"
    if text.startswith("package ") and not text.endswith(";"):
        text += ";"
    return text


def import_score(text: str) -> Tuple[int, int, int]:
    stripped = str(text).strip()
    return (
        stripped.count("."),
        1 if stripped.endswith(";") else 0,
        len(stripped),
    )


def collect_raw_lines(payload: Dict[str, object]) -> Dict[int, Dict[str, object]]:
    merged_lines: Dict[int, Dict[str, object]] = {}
    for page in payload.get("pages", []):
        line_range = page.get("lines") or [1, 1]
        start_line = int(line_range[0]) if line_range else 1
        structured = list(page.get("roi_result", {}).get("structured_lines", []))
        if structured:
            for offset, item in enumerate(structured):
                absolute_line_no = start_line + offset
                merged_lines[absolute_line_no] = {
                    "absolute_line_no": absolute_line_no,
                    "raw_text": str(item.get("text", "")),
                }
            continue
        for roi in page.get("roi_result", {}).get("rois", []):
            if roi.get("name") != "code":
                continue
            for offset, item in enumerate(roi.get("lines", [])):
                absolute_line_no = start_line + offset
                merged_lines[absolute_line_no] = {
                    "absolute_line_no": absolute_line_no,
                    "raw_text": str(item.get("text", "")),
                }
            break
    return merged_lines


def collect_cleaned_lines(payload: Dict[str, object]) -> Dict[int, Dict[str, object]]:
    merged: Dict[int, Dict[str, object]] = {}
    for item in payload.get("merged_lines", []):
        line_no = int(item.get("absolute_line_no") or item.get("line_no") or item.get("index") or 0)
        if line_no <= 0:
            continue
        merged[line_no] = dict(item)
    return merged


def choose_line(raw_text: str, cleaned_text: str) -> str:
    raw_norm = normalize_light(raw_text)
    cleaned_norm = replace_cjk_punctuation(str(cleaned_text)).strip()
    raw_kind = classify_line(raw_norm)
    cleaned_kind = classify_line(cleaned_norm)

    if raw_kind in {"import", "package"}:
        raw_norm = normalize_import_or_package(raw_norm)
    if cleaned_kind in {"import", "package"}:
        cleaned_norm = normalize_import_or_package(cleaned_norm)

    if cleaned_kind == "blank" and raw_kind != "blank":
        return raw_norm
    if raw_kind == "import" and cleaned_kind != "import":
        return raw_norm
    if raw_kind == "package" and cleaned_kind != "package":
        return raw_norm
    if raw_kind == "comment" and cleaned_kind in {"blank", "comment"} and len(raw_norm) > len(cleaned_norm):
        return raw_norm
    if raw_kind == "import" and cleaned_kind == "import":
        return raw_norm if import_score(raw_norm) >= import_score(cleaned_norm) else cleaned_norm
    if cleaned_norm:
        return cleaned_norm
    return raw_norm


def build_fused_text(payload: Dict[str, object]) -> str:
    raw_by_line = collect_raw_lines(payload)
    cleaned_by_line = collect_cleaned_lines(payload)
    max_line_no = max(list(raw_by_line.keys()) + list(cleaned_by_line.keys()) + [0])
    lines: List[str] = []
    for line_no in range(1, max_line_no + 1):
        raw_text = str(raw_by_line.get(line_no, {}).get("raw_text", ""))
        cleaned_text = str(cleaned_by_line.get(line_no, {}).get("cleaned_text", ""))
        lines.append(choose_line(raw_text, cleaned_text))
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    ocr_root = os.path.abspath(args.ocr_root)
    output_root = os.path.abspath(args.output_root)
    for path in iter_ocr_files(ocr_root):
        payload = load_json(path)
        out_path = normalize_path_output(ocr_root, output_root, path)
        write_text(out_path, build_fused_text(payload))
        print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
