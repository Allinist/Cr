#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List


FENCE_LINE_RE = re.compile(r"^\s*```[A-Za-z0-9_-]*\s*$")
FENCE_INLINE_RE = re.compile(r"```")
HALLUCINATION_PREFIXES = (
    "The following is a sample of text extracted from a document.",
    "The image contains a single block of text",
    "The image contains",
)
HALLUCINATION_SNIPPETS = (
    "The following is a sample of text extracted from a document. It is not intended to be a complete transcription of the entire document.",
)
HALLUCINATION_EXACT = {"---", "them."}
CJK_PUNCT_TRANSLATION = str.maketrans(
    {
        "，": ",",
        "。": ".",
        "；": ";",
        "：": ":",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "［": "[",
        "］": "]",
        "｛": "{",
        "｝": "}",
        "《": "<",
        "》": ">",
        "〈": "<",
        "〉": ">",
        "“": "\"",
        "”": "\"",
        "‘": "'",
        "’": "'",
        "、": ",",
        "！": "!",
        "？": "?",
        "＝": "=",
        "＋": "+",
        "－": "-",
        "／": "/",
        "＊": "*",
        "％": "%",
        "｜": "|",
        "＆": "&",
        "＠": "@",
        "＃": "#",
        "＾": "^",
        "～": "~",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a lightly cleaned source file from aggregated OCR JSON.")
    parser.add_argument("--input", required=True, help="Aggregated .ocr.json path.")
    parser.add_argument("--json-output", required=True, help="Path to write lightly cleaned line JSON.")
    parser.add_argument("--code-output", required=True, help="Path to write lightly cleaned source text.")
    return parser.parse_args()


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)


def write_json(path: str, payload: Dict[str, object]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_text(path: str, text: str) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text.rstrip() + "\n")


def replace_cjk_punctuation(text: str) -> str:
    return str(text).translate(CJK_PUNCT_TRANSLATION)


def strip_markdown_artifacts(text: str) -> str:
    value = str(text).replace("\r\n", "\n").replace("\r", "\n")
    parts = [segment.strip() for segment in value.split("\n")]
    kept: List[str] = []
    for part in parts:
        if not part:
            continue
        if FENCE_LINE_RE.match(part):
            continue
        kept.append(part)
    value = " ".join(kept).strip()
    value = FENCE_INLINE_RE.sub("", value).strip()
    return value


def light_clean_text(text: str) -> str:
    value = strip_markdown_artifacts(text)
    if not value:
        return ""
    for snippet in HALLUCINATION_SNIPPETS:
        value = value.replace(snippet, "").strip()
    value = re.sub(r"\s+", " ", value).strip()
    if value in HALLUCINATION_EXACT:
        return ""
    for prefix in HALLUCINATION_PREFIXES:
        if value.startswith(prefix):
            return ""
    return replace_cjk_punctuation(value).strip()


def flatten_lines(payload: Dict[str, object]) -> List[Dict[str, object]]:
    flat: List[Dict[str, object]] = []
    for page in payload.get("pages", []):
        for item in page.get("cleaned_lines", []):
            absolute_line_no = int(item.get("absolute_line_no") or item.get("header_line_no") or 0)
            flat.append(
                {
                    "absolute_line_no": absolute_line_no,
                    "header_line_no": item.get("header_line_no"),
                    "raw_text": str(item.get("raw_text", "")),
                }
            )
    flat.sort(key=lambda item: int(item.get("absolute_line_no") or 0))
    return flat


def build_output_lines(flat_lines: List[Dict[str, object]]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for item in flat_lines:
        cleaned_text = light_clean_text(str(item.get("raw_text", "")))
        output.append(
            {
                "absolute_line_no": int(item.get("absolute_line_no") or 0),
                "header_line_no": item.get("header_line_no"),
                "raw_text": str(item.get("raw_text", "")),
                "cleaned_text": cleaned_text,
            }
        )
    return output


def build_code_text(lines: List[Dict[str, object]]) -> str:
    if not lines:
        return ""
    max_line_no = max(int(item.get("absolute_line_no") or 0) for item in lines)
    by_line = {int(item.get("absolute_line_no") or 0): str(item.get("cleaned_text", "")) for item in lines}
    ordered: List[str] = []
    for line_no in range(1, max_line_no + 1):
        ordered.append(by_line.get(line_no, ""))
    return "\n".join(ordered)


def main() -> int:
    args = parse_args()
    with open(args.input, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    flat_lines = flatten_lines(payload)
    output_lines = build_output_lines(flat_lines)
    code_text = build_code_text(output_lines)
    write_json(
        args.json_output,
        {
            "source": os.path.abspath(args.input),
            "file": payload.get("file"),
            "page_total": payload.get("page_total"),
            "line_count": len(output_lines),
            "lines": output_lines,
        },
    )
    write_text(args.code_output, code_text)
    print(args.json_output)
    print(args.code_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
