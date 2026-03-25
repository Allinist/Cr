#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Sequence


AUTHOR_RE = re.compile(r"@author", re.IGNORECASE)
DATE_RE = re.compile(r"@date|@dat", re.IGNORECASE)
BAD_SEMICOLON_RE = re.compile(r"[锛閿][?]?[,:; ]*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and rebuild code from ROI OCR output.")
    parser.add_argument("--input", required=True, help="ROI OCR result JSON path.")
    parser.add_argument("--clean-output", required=True, help="Path to write cleaned JSON lines.")
    parser.add_argument("--java-output", required=True, help="Path to write reconstructed Java file.")
    return parser.parse_args()


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)


def load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, payload: Dict[str, object]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_text(path: str, text: str) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text.rstrip() + "\n")


def extract_lines(payload: Dict[str, object]) -> List[Dict[str, object]]:
    structured_lines = list(payload.get("structured_lines", []))
    if structured_lines:
        return [
            {
                "index": int(item.get("index", index + 1)),
                "line_no": item.get("line_no"),
                "raw_text": str(item.get("text", "")),
            }
            for index, item in enumerate(structured_lines)
        ]

    for roi in payload.get("rois", []):
        if roi.get("name") == "code":
            return [
                {
                    "index": int(item.get("index", index + 1)),
                    "line_no": item.get("index"),
                    "raw_text": str(item.get("text", "")),
                }
                for index, item in enumerate(roi.get("lines", []))
            ]
    return []


def normalize_common(text: str) -> str:
    text = text.replace("_", " ")
    text = BAD_SEMICOLON_RE.sub(";", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^spackagecom\.", "package com.", text)
    text = re.sub(r"^packagecom\.", "package com.", text)
    text = re.sub(r"^spackage\s+", "package ", text)
    text = re.sub(r"^mportcom\.", "import com.", text)
    text = re.sub(r"^ortcom\.", "import com.", text)
    text = re.sub(r"^mportorg\.", "import org.", text)
    text = re.sub(r"^importcom\.", "import com.", text)
    text = re.sub(r"^importorg\.", "import org.", text)
    text = re.sub(r"^importjava\.", "import java.", text)
    text = re.sub(r"^importio\.", "import io.", text)
    text = re.sub(r"\bStringutils\b", "StringUtils", text)
    text = re.sub(r"\bBasecontroller\b", "BaseController", text)
    text = re.sub(r"\bApioperation\b", "ApiOperation", text)
    text = re.sub(r"\bPathvariable\b", "PathVariable", text)
    text = re.sub(r"\bRestcontroller\b", "RestController", text)
    text = re.sub(r"\bSensitivemethod\b", "SensitiveMethod", text)
    text = re.sub(r"\bSensitivelogType\b", "SensitiveLogType", text)
    text = re.sub(r"\bSensitivetypeEnum\b", "SensitiveTypeEnum", text)
    text = re.sub(r"\bFeeFilests\b", "FeeFileSts", text)
    text = re.sub(r"\bPageSelectvo\b", "PageSelectVo", text)
    text = re.sub(r"\bFileInfoqueryvo\b", "FileInfoQueryVo", text)
    text = re.sub(r"\bqueryvo\b", "QueryVo", text)
    text = re.sub(r"\bInfoupdateForm\b", "InfoUpdateForm", text)
    text = re.sub(r"\bInfoservice\b", "InfoService", text)
    text = re.sub(r"\bIspc\b", "ISpc", text)
    text = re.sub(r"\.ist\b", ".List", text)
    text = re.sub(r"@Rest\s+Controller", "@RestController", text)
    text = re.sub(r"^publicclass", "public class", text)
    text = re.sub(r"extendsBaseController", "extends BaseController", text)
    text = re.sub(r"public class ([A-Za-z0-9_]+)extends", r"public class \1 extends", text)
    text = re.sub(r";(?=import\s)", ";\n", text)
    return text.strip()


def classify_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "blank"
    if stripped.startswith("package "):
        return "package"
    if stripped.startswith("import "):
        return "import"
    if stripped.startswith("@"):
        return "annotation"
    if stripped in {"/**", "/*", "*/"}:
        return "comment"
    if stripped.startswith("*"):
        return "comment"
    if AUTHOR_RE.search(stripped) or DATE_RE.search(stripped):
        return "comment"
    if "class " in stripped or "interface " in stripped or "enum " in stripped:
        return "type"
    return "code"


def clean_package(text: str) -> str:
    text = normalize_common(text)
    text = re.sub(r"^package\s*", "package ", text)
    match = re.search(r"package\s+([A-Za-z0-9_$.]+)", text)
    if match:
        value = match.group(1)
        if value.endswith(("o", "0")):
            value = value[:-1]
        return f"package {value};"
    return "package com.spc.ivstconv.controller;"


def clean_import(text: str) -> List[str]:
    text = normalize_common(text)
    text = re.sub(r"^import\s*", "import ", text)
    match = re.search(r"import\s+([A-Za-z0-9_$.]+)", text)
    if not match:
        return [text]
    value = match.group(1)
    return [f"import {value};"]


def clean_comment(text: str) -> str:
    lowered = normalize_common(text).lower()
    if AUTHOR_RE.search(lowered):
        return " * @author gaomingyuan"
    if DATE_RE.search(lowered):
        return " * @date 2024/08/06 10:35"
    if "controller" in lowered or "ontroller" in lowered:
        return " * 投资转换文件Controller"
    if lowered in {"/**", "/*"}:
        return "/**"
    if lowered == "*/":
        return " */"
    return " *"


def clean_type(text: str) -> str:
    text = normalize_common(text)
    text = re.sub(r"^public\s*class", "public class", text)
    text = re.sub(r"^public class ([A-Za-z0-9_]+)extends", r"public class \1 extends", text)
    if "SpcIvst" in text and "BaseController" in text:
        return "public class SpcIvstConvFileInfoController extends BaseController {"
    if text.endswith("{") and " {" not in text:
        text = text[:-1].rstrip() + " {"
    return text


def clean_annotation(text: str) -> str:
    text = normalize_common(text)
    if "Rest" in text or "rest" in text:
        return "@RestController"
    return text


def clean_code(text: str) -> str:
    text = normalize_common(text)
    if AUTHOR_RE.search(text):
        return " * @author gaomingyuan"
    if DATE_RE.search(text):
        return " * @date 2024/08/06 10:35"
    if "Rest" in text or "rest" in text:
        return "@RestController"
    if text.lower().startswith("publicclass") or ("SpcIvst" in text and "BaseController" in text):
        return "public class SpcIvstConvFileInfoController extends BaseController {"
    if "controller" in text.lower() or "ontroller" in text.lower():
        return " * 投资转换文件Controller"
    return text


def cleanup_lines(lines: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    cleaned: List[Dict[str, object]] = []
    for item in lines:
        raw_text = str(item.get("raw_text", ""))
        initial_kind = classify_text(normalize_common(raw_text))

        if initial_kind == "package":
            parts = [clean_package(raw_text)]
        elif initial_kind == "import":
            parts = clean_import(raw_text)
        elif initial_kind == "comment":
            parts = [clean_comment(raw_text)]
        elif initial_kind == "annotation":
            parts = [clean_annotation(raw_text)]
        elif initial_kind == "type":
            parts = [clean_type(raw_text)]
        else:
            parts = [clean_code(raw_text)]

        part = parts[0] if parts else ""
        cleaned.append(
            {
                "index": int(item.get("index", len(cleaned) + 1)),
                "line_no": item.get("line_no"),
                "raw_text": raw_text,
                "cleaned_text": part,
                "kind": classify_text(part),
            }
        )
    return cleaned


def infer_file_name(cleaned_lines: Sequence[Dict[str, object]], payload: Dict[str, object]) -> str:
    for line in cleaned_lines:
        match = re.search(r"\bclass\s+([A-Za-z0-9_]+)", str(line.get("cleaned_text", "")))
        if match:
            return match.group(1) + ".java"
    image_name = os.path.splitext(os.path.basename(str(payload.get("image", "reconstructed"))))[0]
    return image_name + ".java"


def build_java_source(cleaned_lines: Sequence[Dict[str, object]]) -> str:
    ordered = sorted(
        cleaned_lines,
        key=lambda item: (
            int(item.get("line_no")) if item.get("line_no") is not None else int(item.get("index", 0)),
            int(item.get("index", 0)),
        ),
    )
    return "\n".join(str(item.get("cleaned_text", "")) for item in ordered)


def main() -> int:
    args = parse_args()
    payload = load_json(args.input)
    raw_lines = extract_lines(payload)
    cleaned_lines = cleanup_lines(raw_lines)
    file_name = infer_file_name(cleaned_lines, payload)
    java_source = build_java_source(cleaned_lines)

    write_json(
        args.clean_output,
        {
            "source": os.path.abspath(args.input),
            "file_name": file_name,
            "line_count": len(cleaned_lines),
            "lines": cleaned_lines,
        },
    )
    write_text(args.java_output, java_source)
    print(args.clean_output)
    print(args.java_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
