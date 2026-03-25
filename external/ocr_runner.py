#!/usr/bin/env python3
"""
OCR runner for code screenshots.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from rapidocr import RapidOCR
from rapidocr.utils.typings import LangRec, ModelType, OCRVersion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR on a screenshot.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--lang", default="auto", help="Reserved for future model routing.")
    parser.add_argument("--header-ratio", type=float, default=0.18, help="Top area ratio used for page markers and header.")
    parser.add_argument("--footer-ratio", type=float, default=0.06, help="Bottom area ratio used for page end markers.")
    parser.add_argument("--ocr-mode", choices=["fast", "balanced", "quality"], default="balanced", help="OCR quality profile.")
    parser.add_argument("--accel", choices=["auto", "cpu", "directml", "openvino"], default="auto", help="Preferred inference acceleration backend.")
    parser.add_argument("--model-preset", choices=["default", "server", "doc"], default="server", help="OCR model preset.")
    parser.add_argument("--upscale", type=float, default=0.0, help="Override upscale factor; 0 means use the profile default.")
    parser.add_argument("--body-tiles", type=int, default=0, help="Split body into N horizontal tiles for extra OCR passes; 0 means use the profile default.")
    parser.add_argument("--tile-overlap", type=int, default=36, help="Vertical overlap in pixels for body tiles.")
    parser.add_argument("--text-score", type=float, default=0.45, help="Minimum text confidence after OCR post-filtering.")
    parser.add_argument("--box-thresh", type=float, default=0.35, help="Detection box threshold.")
    parser.add_argument("--unclip-ratio", type=float, default=1.8, help="Detection unclip ratio.")
    parser.add_argument("--max-side-len", type=int, default=0, help="Override RapidOCR max side length; 0 means use the profile default.")
    parser.add_argument("--det-limit-side-len", type=int, default=0, help="Override detector side length; 0 means use the profile default.")
    return parser.parse_args()


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def resolve_input_path(path: str) -> str:
    return os.path.abspath(os.path.normpath(path))


def polygon_bounds(polygon: Any) -> Tuple[float, float, float, float]:
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def sort_entries_reading_order(entries: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not entries:
        return entries

    enriched = []
    for entry in entries:
        polygon = entry.get("polygon") or []
        if not polygon:
            enriched.append((0.0, 0.0, 0.0, entry))
            continue
        x1, y1, x2, y2 = polygon_bounds(polygon)
        enriched.append((y1, x1, y2 - y1, entry))

    average_height = sum(item[2] for item in enriched) / max(len(enriched), 1)
    line_threshold = max(8.0, average_height * 0.7)
    enriched.sort(key=lambda item: (item[0], item[1]))

    line_groups: List[List[Tuple[float, float, float, Dict[str, object]]]] = []
    for item in enriched:
        if not line_groups:
            line_groups.append([item])
            continue
        current_group = line_groups[-1]
        group_y = sum(member[0] for member in current_group) / len(current_group)
        if abs(item[0] - group_y) <= line_threshold:
            current_group.append(item)
        else:
            line_groups.append([item])

    sorted_entries: List[Dict[str, object]] = []
    for group in line_groups:
        group.sort(key=lambda item: item[1])
        for _, _, _, entry in group:
            sorted_entries.append(entry)
    return sorted_entries


def ocr_text(entries: List[Dict[str, object]]) -> str:
    return "\n".join(str(item.get("text", "")).strip() for item in entries if str(item.get("text", "")).strip())


def upscale_image(image: np.ndarray, scale: float = 1.8) -> np.ndarray:
    height, width = image.shape[:2]
    return cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_CUBIC)


def preprocess_region(image: np.ndarray, invert: bool = False, sharpen: bool = False) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    if sharpen:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, kernel)
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    binary = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    if invert:
        binary = cv2.bitwise_not(binary)
    return binary


def preprocess_gray(image: np.ndarray, sharpen: bool = False, clahe: bool = False) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if clahe:
        clahe_op = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        gray = clahe_op.apply(gray)
    else:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    if sharpen:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, kernel)
    return gray


def crop_regions(image: np.ndarray, header_ratio: float, footer_ratio: float) -> Dict[str, Tuple[np.ndarray, Dict[str, int]]]:
    height, width = image.shape[:2]
    header_height = max(1, int(height * header_ratio))
    footer_height = max(1, int(height * footer_ratio))
    body_start = header_height
    body_end = max(body_start + 1, height - footer_height)

    body_width = width
    line_number_width = max(72, min(int(body_width * 0.08), 140))

    return {
        "header": (
            image[0:header_height, :],
            {"x": 0, "y": 0, "width": width, "height": header_height},
        ),
        "body": (
            image[body_start:body_end, :],
            {"x": 0, "y": body_start, "width": width, "height": body_end - body_start},
        ),
        "body_line_numbers": (
            image[body_start:body_end, 0:line_number_width],
            {"x": 0, "y": body_start, "width": line_number_width, "height": body_end - body_start},
        ),
        "body_code": (
            image[body_start:body_end, line_number_width:width],
            {"x": line_number_width, "y": body_start, "width": width - line_number_width, "height": body_end - body_start},
        ),
        "footer": (
            image[body_end:height, :],
            {"x": 0, "y": body_end, "width": width, "height": height - body_end},
        ),
    }


def profile_value(mode: str, fast: Any, balanced: Any, quality: Any) -> Any:
    if mode == "fast":
        return fast
    if mode == "quality":
        return quality
    return balanced


def available_acceleration() -> Dict[str, bool]:
    providers = set(ort.get_available_providers())
    openvino_installed = importlib.util.find_spec("openvino") is not None
    return {
        "directml": "DmlExecutionProvider" in providers,
        "openvino": openvino_installed,
    }


def resolve_acceleration(accel: str) -> str:
    available = available_acceleration()
    if accel == "cpu":
        return "cpu"
    if accel == "directml":
        return "directml" if available["directml"] else "cpu"
    if accel == "openvino":
        return "openvino" if available["openvino"] else "cpu"
    if available["directml"]:
        return "directml"
    if available["openvino"]:
        return "openvino"
    return "cpu"


def flatten_rapidocr_result(result: Any) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    if not result:
        return entries

    boxes = getattr(result, "boxes", None)
    txts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    if boxes is None or txts is None or scores is None:
        return entries

    for polygon, text, confidence in zip(boxes, txts, scores):
        entries.append(
            {
                "text": str(text),
                "confidence": float(confidence),
                "polygon": polygon.tolist() if hasattr(polygon, "tolist") else polygon,
            }
        )
    return entries


def image_to_entries(engine: RapidOCR, image: np.ndarray) -> List[Dict[str, object]]:
    result = engine(image)
    return sort_entries_reading_order(flatten_rapidocr_result(result))


def translate_entries(entries: List[Dict[str, object]], offset_x: int = 0, offset_y: int = 0) -> List[Dict[str, object]]:
    translated = []
    for entry in entries:
        polygon = entry.get("polygon") or []
        shifted = []
        for point in polygon:
            shifted.append([float(point[0]) + offset_x, float(point[1]) + offset_y])
        translated.append(
            {
                "text": entry.get("text", ""),
                "confidence": float(entry.get("confidence", 0.0)),
                "polygon": shifted,
            }
        )
    return translated


def entry_key(entry: Dict[str, object]) -> Tuple[int, int]:
    polygon = entry.get("polygon") or []
    if not polygon:
        return (0, 0)
    x1, y1, x2, y2 = polygon_bounds(polygon)
    return (int(round((x1 + x2) / 2 / 10.0)), int(round((y1 + y2) / 2 / 10.0)))


def dedupe_entries(entries: List[Dict[str, object]]) -> List[Dict[str, object]]:
    best_by_key: Dict[Tuple[int, int], Dict[str, object]] = {}
    for entry in entries:
        text = str(entry.get("text", "")).strip()
        if not text:
            continue
        key = entry_key(entry)
        current = best_by_key.get(key)
        current_score = (
            float(current.get("confidence", 0.0)) + min(len(str(current.get("text", ""))), 32) * 0.005
            if current is not None
            else -1.0
        )
        score = float(entry.get("confidence", 0.0)) + min(len(text), 32) * 0.005
        if current is None or score > current_score:
            best_by_key[key] = entry
    return sort_entries_reading_order(list(best_by_key.values()))


def score_header_entries(entries: List[Dict[str, object]]) -> int:
    text = ocr_text(entries).upper()
    score = 0
    for token in ("PAGE-BEGIN", "PAGE-END", "FILE", "RILE", "F1LE", "PAGE", "LINES"):
        if token in text:
            score += 2
    score += sum(1 for entry in entries if len(str(entry.get("text", ""))) > 8)
    return score


def choose_best_entries(engine: RapidOCR, image: np.ndarray, is_header: bool = False) -> List[Dict[str, object]]:
    upscaled = upscale_image(image)
    variants = [
        preprocess_gray(upscaled, sharpen=False, clahe=False),
        preprocess_gray(upscaled, sharpen=True, clahe=True),
        preprocess_region(upscaled, invert=False, sharpen=True),
    ]
    best_entries: List[Dict[str, object]] = []
    best_score = -1
    for variant in variants:
        entries = image_to_entries(engine, variant)
        score = score_header_entries(entries) if is_header else len(entries)
        if score > best_score:
            best_entries = entries
            best_score = score
    return best_entries


def choose_body_entries(
    engine: RapidOCR,
    image: np.ndarray,
    upscale: float,
    tiles: int,
    tile_overlap: int,
    mode: str,
) -> List[Dict[str, object]]:
    upscaled = upscale_image(image, upscale)
    variants = [
        preprocess_gray(upscaled, sharpen=False, clahe=False),
        preprocess_gray(upscaled, sharpen=True, clahe=True),
        preprocess_region(upscaled, invert=False, sharpen=True),
    ]
    if mode == "quality":
        variants.append(preprocess_region(upscaled, invert=True, sharpen=False))
    elif mode == "fast":
        variants = variants[:2]
    merged_entries: List[Dict[str, object]] = []
    for variant in variants:
        merged_entries.extend(image_to_entries(engine, variant))

    if tiles > 1:
        height = image.shape[0]
        step = max(1, int(np.ceil(height / float(tiles))))
        for start_y in range(0, height, step):
            end_y = min(height, start_y + step + tile_overlap)
            tile = image[start_y:end_y, :]
            if tile.size == 0:
                continue
            upscaled_tile = upscale_image(tile, upscale)
            tile_variants = [
                preprocess_gray(upscaled_tile, sharpen=False, clahe=False),
                preprocess_region(upscaled_tile, invert=False, sharpen=True),
            ]
            if mode == "quality":
                tile_variants.append(preprocess_gray(upscaled_tile, sharpen=True, clahe=True))
            for variant in tile_variants:
                tile_entries = image_to_entries(engine, variant)
                scale_y = max(1e-6, float(tile.shape[0]) / float(variant.shape[0]))
                scale_x = max(1e-6, float(tile.shape[1]) / float(variant.shape[1]))
                normalized_entries = []
                for entry in tile_entries:
                    polygon = []
                    for point in entry.get("polygon") or []:
                        polygon.append([float(point[0]) * scale_x, float(point[1]) * scale_y])
                    normalized_entries.append(
                        {
                            "text": entry.get("text", ""),
                            "confidence": float(entry.get("confidence", 0.0)),
                            "polygon": polygon,
                        }
                    )
                merged_entries.extend(translate_entries(normalized_entries, offset_y=start_y))
    return dedupe_entries(merged_entries)


def build_engine(args: argparse.Namespace) -> RapidOCR:
    mode = args.ocr_mode
    resolved_accel = resolve_acceleration(args.accel)
    is_heavy_model = args.model_preset in {"server", "doc"}
    if resolved_accel == "directml" and is_heavy_model:
        max_side_len = args.max_side_len or profile_value(mode, 1800, 2200, 2600)
        det_limit_side_len = args.det_limit_side_len or profile_value(mode, 736, 960, 1120)
        rec_batch_num = profile_value(mode, 2, 3, 4)
    else:
        max_side_len = args.max_side_len or profile_value(mode, 2000, 2800, 3600)
        det_limit_side_len = args.det_limit_side_len or profile_value(mode, 736, 1120, 1400)
        rec_batch_num = profile_value(mode, 8, 12, 16) if resolved_accel == "directml" else profile_value(mode, 6, 8, 10)
    params = {
        "Global.text_score": args.text_score,
        "Global.max_side_len": max_side_len,
        "Det.limit_side_len": det_limit_side_len,
        "Det.box_thresh": args.box_thresh,
        "Det.unclip_ratio": args.unclip_ratio,
        "Rec.rec_batch_num": rec_batch_num,
        "EngineConfig.onnxruntime.enable_cpu_mem_arena": True,
        "EngineConfig.onnxruntime.intra_op_num_threads": max(1, (os.cpu_count() or 4) - 1),
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
    }
    if args.model_preset == "server":
        params.update(
            {
                "Det.model_type": ModelType.SERVER,
                "Rec.model_type": ModelType.SERVER,
                "Det.ocr_version": OCRVersion.PPOCRV4,
                "Rec.ocr_version": OCRVersion.PPOCRV4,
                "Rec.lang_type": LangRec.CH,
                "Rec.rec_img_shape": [3, 48, 320] if resolved_accel == "directml" else [3, 48, 512],
            }
        )
    elif args.model_preset == "doc":
        params.update(
            {
                "Det.model_type": ModelType.SERVER,
                "Rec.model_type": ModelType.SERVER,
                "Det.ocr_version": OCRVersion.PPOCRV4,
                "Rec.ocr_version": OCRVersion.PPOCRV4,
                "Rec.lang_type": LangRec.CH_DOC,
                "Rec.rec_img_shape": [3, 48, 320] if resolved_accel == "directml" else [3, 48, 512],
            }
        )
    if resolved_accel == "directml":
        params.update(
            {
                "EngineConfig.onnxruntime.use_dml": True,
            }
        )
    return RapidOCR(params=params)


def main() -> int:
    args = parse_args()
    engine = build_engine(args)
    image_path = resolve_input_path(args.image)
    source_image = cv2.imread(image_path)
    if source_image is None:
        raise RuntimeError(
            "unable to read image: %s (resolved: %s). Use ./screenshots/test.png or an absolute path."
            % (args.image, image_path)
        )

    regions = crop_regions(source_image, args.header_ratio, args.footer_ratio)
    header_image, header_box = regions["header"]
    body_image, body_box = regions["body"]
    body_line_numbers_image, body_line_numbers_box = regions["body_line_numbers"]
    body_code_image, body_code_box = regions["body_code"]
    footer_image, footer_box = regions["footer"]

    if resolve_acceleration(args.accel) == "directml" and args.model_preset in {"server", "doc"}:
        upscale = args.upscale or profile_value(args.ocr_mode, 1.35, 1.55, 1.7)
        body_tiles = args.body_tiles or 1
    else:
        upscale = args.upscale or profile_value(args.ocr_mode, 1.45, 1.85, 2.0)
        body_tiles = args.body_tiles or profile_value(args.ocr_mode, 1, 1, 2)

    header_entries = choose_best_entries(engine, header_image, is_header=True)
    body_entries = choose_body_entries(engine, body_image, upscale, body_tiles, args.tile_overlap, args.ocr_mode)
    body_line_number_entries = choose_body_entries(engine, body_line_numbers_image, upscale, 1, args.tile_overlap, "fast")
    body_code_entries = choose_body_entries(engine, body_code_image, upscale, body_tiles, args.tile_overlap, args.ocr_mode)
    footer_entries = choose_best_entries(engine, footer_image, is_header=True)

    payload = {
        "image": image_path,
        "engine": "rapidocr",
        "ocr_mode": args.ocr_mode,
        "accel": resolve_acceleration(args.accel),
        "model_preset": args.model_preset,
        "entries": header_entries + body_entries + footer_entries,
        "regions": {
            "header": {
                "box": header_box,
                "entries": header_entries,
                "text": ocr_text(header_entries),
            },
            "body": {
                "box": body_box,
                "entries": body_entries,
                "text": ocr_text(body_entries),
            },
            "body_line_numbers": {
                "box": body_line_numbers_box,
                "entries": body_line_number_entries,
                "text": ocr_text(body_line_number_entries),
            },
            "body_code": {
                "box": body_code_box,
                "entries": body_code_entries,
                "text": ocr_text(body_code_entries),
            },
            "footer": {
                "box": footer_box,
                "entries": footer_entries,
                "text": ocr_text(footer_entries),
            },
        },
    }
    ensure_parent(args.output)
    with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
