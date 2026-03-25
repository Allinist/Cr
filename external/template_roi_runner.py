#!/usr/bin/env python3
"""
Experimental OCR pipeline:
- template alignment against a reference screenshot
- fixed ROI extraction
- line segmentation inside ROIs
- CRNN-CTC ONNX recognition with DirectML fallback

This file is intentionally isolated from the main RapidOCR flow so we can
experiment without destabilizing the default path.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from rapidocr import RapidOCR
from rapidocr.utils.typings import LangRec, ModelType, OCRVersion


@dataclass
class Roi:
    name: str
    x: int
    y: int
    width: int
    height: int


@dataclass
class LineBox:
    top: int
    bottom: int
    left: int
    right: int

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental template-aligned ROI OCR runner.")
    parser.add_argument("--image", required=True, help="Captured screenshot path.")
    parser.add_argument("--layout", required=True, help="Layout JSON with reference image and ROI definitions.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--debug-dir", default=None, help="Optional directory to save aligned ROI crops.")
    return parser.parse_args()


def load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, payload: Dict[str, object]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def resolve_layout_path(base_path: str, maybe_relative: str) -> str:
    if os.path.isabs(maybe_relative):
        return maybe_relative
    return os.path.abspath(os.path.join(os.path.dirname(base_path), "..", maybe_relative))


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise RuntimeError("unable to read image: %s" % path)
    return image


def build_rois(layout: Dict[str, object]) -> List[Roi]:
    rois = []
    for item in layout.get("rois", []):
        rois.append(
            Roi(
                name=str(item["name"]),
                x=int(item["x"]),
                y=int(item["y"]),
                width=int(item["width"]),
                height=int(item["height"]),
            )
        )
    return rois


def align_to_reference(reference: np.ndarray, image: np.ndarray, layout: Dict[str, object]) -> np.ndarray:
    alignment = layout.get("alignment", {})
    max_features = int(alignment.get("max_features", 1500))
    keep_percent = float(alignment.get("keep_percent", 0.2))
    min_match_count = int(alignment.get("min_match_count", 12))

    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(max_features)
    ref_kp, ref_desc = orb.detectAndCompute(ref_gray, None)
    img_kp, img_desc = orb.detectAndCompute(img_gray, None)
    if ref_desc is None or img_desc is None:
        return image

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = list(matcher.match(img_desc, ref_desc))
    if len(matches) < min_match_count:
        return image
    matches.sort(key=lambda item: item.distance)
    keep_count = max(min_match_count, int(len(matches) * keep_percent))
    matches = matches[:keep_count]

    src = np.float32([img_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([ref_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography, _ = cv2.findHomography(src, dst, cv2.RANSAC)
    if homography is None:
        return image
    return cv2.warpPerspective(image, homography, (reference.shape[1], reference.shape[0]))


def crop_roi(image: np.ndarray, roi: Roi) -> np.ndarray:
    x1 = max(0, roi.x)
    y1 = max(0, roi.y)
    x2 = min(image.shape[1], roi.x + roi.width)
    y2 = min(image.shape[0], roi.y + roi.height)
    return image[y1:y2, x1:x2]


def resolve_providers(preferred: Sequence[str]) -> List[str]:
    available = set(ort.get_available_providers())
    selected = [provider for provider in preferred if provider in available]
    if "CPUExecutionProvider" not in selected and "CPUExecutionProvider" in available:
        selected.append("CPUExecutionProvider")
    if not selected:
        selected = ["CPUExecutionProvider"]
    return selected


def load_charset(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def preprocess_for_segmentation(image: np.ndarray, invert: bool = False) -> np.ndarray:
    gray = to_gray(image)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV if not invert else cv2.THRESH_BINARY,
        31,
        11,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def connected_components_line_boxes(image: np.ndarray, segmentation_cfg: Dict[str, object]) -> List[LineBox]:
    binary = preprocess_for_segmentation(image)
    min_component_area = int(segmentation_cfg.get("min_component_area", 20))
    min_component_height = int(segmentation_cfg.get("min_component_height", 12))
    merge_gap = int(segmentation_cfg.get("merge_gap", 6))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    candidates: List[Tuple[int, int, int, int]] = []
    for label_index in range(1, num_labels):
        x = int(stats[label_index, cv2.CC_STAT_LEFT])
        y = int(stats[label_index, cv2.CC_STAT_TOP])
        width = int(stats[label_index, cv2.CC_STAT_WIDTH])
        height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area < min_component_area or height < min_component_height:
            continue
        candidates.append((y, y + height, x, x + width))

    candidates.sort(key=lambda item: item[0])
    merged: List[LineBox] = []
    for top, bottom, left, right in candidates:
        if not merged:
            merged.append(LineBox(top=top, bottom=bottom, left=left, right=right))
            continue
        previous = merged[-1]
        if top - previous.bottom <= merge_gap:
            merged[-1] = LineBox(
                top=min(previous.top, top),
                bottom=max(previous.bottom, bottom),
                left=min(previous.left, left),
                right=max(previous.right, right),
            )
        else:
            merged.append(LineBox(top=top, bottom=bottom, left=left, right=right))
    return merged


def is_reasonable_line_boxes(
    line_boxes: Sequence[LineBox],
    image_height: int,
    segmentation_cfg: Dict[str, object],
) -> bool:
    if not line_boxes:
        return False
    min_lines = int(segmentation_cfg.get("min_anchor_lines", 8))
    max_lines = int(segmentation_cfg.get("max_anchor_lines", 120))
    if len(line_boxes) < min_lines or len(line_boxes) > max_lines:
        return False
    heights = sorted(box.height for box in line_boxes if box.height > 0)
    if not heights:
        return False
    median_height = heights[len(heights) // 2]
    min_height = int(segmentation_cfg.get("min_anchor_height", 12))
    max_height = int(segmentation_cfg.get("max_anchor_height", max(24, image_height // 8)))
    return min_height <= median_height <= max_height


def projection_segments(values: np.ndarray, threshold: float, min_length: int) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    start = None
    for index, value in enumerate(values):
        active = float(value) >= threshold
        if active and start is None:
            start = index
        elif not active and start is not None:
            if index - start >= min_length:
                segments.append((start, index))
            start = None
    if start is not None and len(values) - start >= min_length:
        segments.append((start, len(values)))
    return segments


def merge_close_segments(segments: List[Tuple[int, int]], max_gap: int) -> List[Tuple[int, int]]:
    if not segments:
        return []
    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def build_fixed_grid_line_boxes(
    image: np.ndarray,
    segmentation_cfg: Dict[str, object],
    roi_name: str,
) -> List[LineBox]:
    line_count_key = "%s_expected_line_count" % roi_name
    expected_line_count = int(segmentation_cfg.get(line_count_key, segmentation_cfg.get("expected_line_count", 0)))
    if expected_line_count <= 0:
        return []
    top_inset = int(segmentation_cfg.get("%s_top_inset" % roi_name, segmentation_cfg.get("grid_top_inset", 0)))
    bottom_inset = int(segmentation_cfg.get("%s_bottom_inset" % roi_name, segmentation_cfg.get("grid_bottom_inset", 0)))
    left_inset = int(segmentation_cfg.get("%s_left_inset" % roi_name, segmentation_cfg.get("grid_left_inset", 0)))
    right_inset = int(segmentation_cfg.get("%s_right_inset" % roi_name, segmentation_cfg.get("grid_right_inset", 0)))

    usable_top = max(0, top_inset)
    usable_bottom = max(usable_top + 1, image.shape[0] - max(0, bottom_inset))
    usable_left = max(0, left_inset)
    usable_right = max(usable_left + 1, image.shape[1] - max(0, right_inset))
    usable_height = usable_bottom - usable_top
    line_height = float(usable_height) / float(expected_line_count)
    line_height_scale = float(segmentation_cfg.get("%s_line_height_scale" % roi_name, segmentation_cfg.get("line_height_scale", 1.0)))
    center_squeeze_px = float(segmentation_cfg.get("%s_center_squeeze_px" % roi_name, segmentation_cfg.get("center_squeeze_px", 0.0)))
    start_shift_px = float(segmentation_cfg.get("%s_start_shift_px" % roi_name, segmentation_cfg.get("start_shift_px", 0.0)))
    scaled_height = max(1.0, line_height * max(0.5, min(1.5, line_height_scale)))
    center_index = (expected_line_count - 1) / 2.0

    line_boxes: List[LineBox] = []
    for index in range(expected_line_count):
        base_center = usable_top + (index + 0.5) * line_height + start_shift_px
        if center_index > 0:
            normalized = (index - center_index) / center_index
        else:
            normalized = 0.0
        center_offset = -normalized * center_squeeze_px
        center = base_center + center_offset
        top = int(round(center - scaled_height / 2.0))
        bottom = int(round(center + scaled_height / 2.0))
        top = max(usable_top, top)
        bottom = min(usable_bottom, max(top + 1, bottom))
        line_boxes.append(LineBox(top=top, bottom=bottom, left=usable_left, right=usable_right))
    return line_boxes


def uses_fixed_grid(segmentation_cfg: Dict[str, object], roi_name: str) -> bool:
    line_count_key = "%s_expected_line_count" % roi_name
    expected_line_count = int(segmentation_cfg.get(line_count_key, segmentation_cfg.get("expected_line_count", 0)))
    return expected_line_count > 0


def detect_line_boxes(image: np.ndarray, segmentation_cfg: Dict[str, object], roi_name: str) -> List[LineBox]:
    fixed_grid_boxes = build_fixed_grid_line_boxes(image, segmentation_cfg, roi_name)
    if fixed_grid_boxes:
        return fixed_grid_boxes

    min_line_height = int(segmentation_cfg.get("min_line_height", 18))
    merge_gap = int(segmentation_cfg.get("merge_gap", 6))
    row_fraction_key = "line_number_row_threshold_fraction" if roi_name == "line_numbers" else "row_threshold_fraction"
    col_fraction_key = "line_number_col_threshold_fraction" if roi_name == "line_numbers" else "col_threshold_fraction"
    row_fraction = float(segmentation_cfg.get(row_fraction_key, segmentation_cfg.get("row_threshold_fraction", 0.08)))
    col_fraction = float(segmentation_cfg.get(col_fraction_key, segmentation_cfg.get("col_threshold_fraction", 0.03)))
    padding_y = int(segmentation_cfg.get("line_padding_y", 2))
    padding_x = int(segmentation_cfg.get("line_padding_x", 2))

    binary = preprocess_for_segmentation(image)
    row_projection = binary.sum(axis=1) / 255.0
    row_threshold = max(3.0, float(binary.shape[1]) * row_fraction)
    row_segments = merge_close_segments(
        projection_segments(row_projection, row_threshold, min_line_height),
        merge_gap,
    )

    line_boxes: List[LineBox] = []
    for top, bottom in row_segments:
        band = binary[top:bottom, :]
        col_projection = band.sum(axis=0) / 255.0
        col_threshold = max(2.0, float(band.shape[0]) * col_fraction)
        col_segments = projection_segments(col_projection, col_threshold, 4)
        if col_segments:
            left = max(0, col_segments[0][0] - padding_x)
            right = min(band.shape[1], col_segments[-1][1] + padding_x)
        else:
            left = 0
            right = band.shape[1]
        if roi_name == "line_numbers":
            left = 0
            right = band.shape[1]
        line_boxes.append(
            LineBox(
                top=max(0, top - padding_y),
                bottom=min(image.shape[0], bottom + padding_y),
                left=left,
                right=right,
            )
        )
    return line_boxes


def build_code_line_boxes_from_anchors(code_image: np.ndarray, anchors: Sequence[LineBox], padding: int = 6) -> List[LineBox]:
    if not anchors:
        return []
    centers = [anchor.center_y for anchor in anchors]
    boundaries: List[Tuple[int, int]] = []
    for index, center in enumerate(centers):
        if index == 0:
            top = max(0, int(round(center - (centers[index + 1] - center) / 2.0)) - padding) if len(centers) > 1 else max(0, int(center) - 18)
        else:
            top = int(round((centers[index - 1] + center) / 2.0)) - padding
        if index == len(centers) - 1:
            bottom = min(code_image.shape[0], int(round(center + (center - centers[index - 1]) / 2.0)) + padding) if len(centers) > 1 else min(code_image.shape[0], int(center) + 18)
        else:
            bottom = int(round((center + centers[index + 1]) / 2.0)) + padding
        top = max(0, top)
        bottom = min(code_image.shape[0], max(top + 1, bottom))
        boundaries.append((top, bottom))
    return [LineBox(top=top, bottom=bottom, left=0, right=code_image.shape[1]) for top, bottom in boundaries]


def line_boxes_from_number_lines(number_lines: Sequence[Dict[str, object]], image_width: int) -> List[LineBox]:
    anchors: List[LineBox] = []
    for item in number_lines:
        box = item.get("box")
        if not isinstance(box, dict):
            continue
        anchors.append(
            LineBox(
                top=int(box["top"]),
                bottom=int(box["bottom"]),
                left=0,
                right=image_width,
            )
        )
    return anchors


class CrnnCtcRecognizer:
    def __init__(self, model_path: str, charset_path: str, providers: Sequence[str], input_height: int, input_width: int):
        self.session = ort.InferenceSession(model_path, providers=resolve_providers(providers))
        self.charset = load_charset(charset_path)
        self.input_height = input_height
        self.input_width = input_width
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        gray = to_gray(image)
        resized = cv2.resize(gray, (self.input_width, self.input_height))
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - 0.5) / 0.5
        return normalized[np.newaxis, np.newaxis, :, :]

    def decode(self, logits: np.ndarray) -> str:
        indices = np.argmax(logits, axis=-1).tolist()
        result: List[str] = []
        previous = None
        blank_index = len(self.charset)
        for index in indices:
            if index == blank_index or index == previous:
                previous = index
                continue
            if 0 <= index < len(self.charset):
                result.append(self.charset[index])
            previous = index
        return "".join(result)

    def recognize_batch(self, images: Sequence[np.ndarray]) -> List[str]:
        if not images:
            return []
        batches = [self.preprocess(image) for image in images]
        input_tensor = np.concatenate(batches, axis=0)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        logits = outputs[0]
        if logits.ndim == 3:
            sequences = logits
        elif logits.ndim == 4:
            sequences = logits[:, 0, :, :]
        else:
            raise RuntimeError("unexpected CRNN output shape: %s" % (logits.shape,))
        return [self.decode(sequence) for sequence in sequences]


class NullRecognizer:
    def __init__(self):
        self.providers = ["CPUExecutionProvider"]

    def recognize_batch(self, images: Sequence[np.ndarray]) -> List[str]:
        return ["" for _ in images]


class RapidOcrLineRecognizer:
    def __init__(self, cfg: Dict[str, object], layout_path: str):
        model_path = resolve_layout_path(layout_path, str(cfg["model_path"]))
        params = {
            "Global.use_det": False,
            "Global.use_cls": bool(cfg.get("use_cls", False)),
            "Global.use_rec": True,
            "Global.text_score": float(cfg.get("text_score", 0.2)),
            "Rec.model_path": model_path,
            "Rec.rec_batch_num": int(cfg.get("batch_size", 1)),
            "Rec.rec_img_shape": [
                3,
                int(cfg.get("input_height", 48)),
                int(cfg.get("input_width", 320)),
            ],
            "Rec.ocr_version": OCRVersion.PPOCRV4 if str(cfg.get("ocr_version", "PP-OCRv4")) == "PP-OCRv4" else OCRVersion.PPOCRV5,
            "Rec.model_type": ModelType.SERVER if str(cfg.get("model_type", "server")).lower() == "server" else ModelType.MOBILE,
            "Rec.lang_type": LangRec(str(cfg.get("lang_type", "ch"))),
            "EngineConfig.onnxruntime.use_dml": "DmlExecutionProvider" in resolve_providers(cfg.get("providers", ["DmlExecutionProvider", "CPUExecutionProvider"])),
        }
        rec_keys_path = cfg.get("charset_path")
        if rec_keys_path:
            resolved_charset = resolve_layout_path(layout_path, str(rec_keys_path))
            if os.path.exists(resolved_charset):
                params["Rec.rec_keys_path"] = resolved_charset
        self.engine = RapidOCR(params=params)
        self.providers = resolve_providers(cfg.get("providers", ["DmlExecutionProvider", "CPUExecutionProvider"]))

    def recognize_batch(self, images: Sequence[np.ndarray]) -> List[str]:
        results = []
        for image in images:
            output = self.engine(image, use_det=False, use_cls=False, use_rec=True)
            txts = getattr(output, "txts", None)
            if txts:
                results.append(str(txts[0]))
            else:
                results.append("")
        return results


def load_recognizer(layout_path: str, recognizer_cfg: Dict[str, object]) -> Tuple[object, bool, List[str]]:
    recognizer_type = str(recognizer_cfg.get("type", "rapidocr_rec")).lower()
    model_path = resolve_layout_path(layout_path, str(recognizer_cfg["model_path"]))

    if recognizer_type == "rapidocr_rec":
        if not os.path.exists(model_path):
            return NullRecognizer(), False, [model_path]
        recognizer = RapidOcrLineRecognizer(recognizer_cfg, layout_path)
        return recognizer, True, []

    charset_path = resolve_layout_path(layout_path, str(recognizer_cfg["charset_path"]))
    if not os.path.exists(model_path) or not os.path.exists(charset_path):
        recognizer = NullRecognizer()
        missing = []
        if not os.path.exists(model_path):
            missing.append(model_path)
        if not os.path.exists(charset_path):
            missing.append(charset_path)
        return recognizer, False, missing

    recognizer = CrnnCtcRecognizer(
        model_path=model_path,
        charset_path=charset_path,
        providers=list(recognizer_cfg.get("providers", ["DmlExecutionProvider", "CPUExecutionProvider"])),
        input_height=int(recognizer_cfg.get("input_height", 48)),
        input_width=int(recognizer_cfg.get("input_width", 320)),
    )
    return recognizer, True, []


def crop_line(image: np.ndarray, line_box: LineBox) -> np.ndarray:
    return image[line_box.top:line_box.bottom, line_box.left:line_box.right]


def save_debug_image(path: str, image: np.ndarray) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    cv2.imwrite(path, image)


def save_debug_crop(debug_dir: str, roi: Roi, crop: np.ndarray) -> None:
    if roi.name == "line_numbers":
        return
    save_debug_image(os.path.join(debug_dir, "%s.png" % roi.name), crop)


def save_debug_lines(debug_dir: str, roi_name: str, image: np.ndarray, line_boxes: Sequence[LineBox]) -> None:
    if roi_name == "line_numbers":
        return
    roi_dir = os.path.join(debug_dir, roi_name)
    os.makedirs(roi_dir, exist_ok=True)
    for index, line_box in enumerate(line_boxes, start=1):
        line_image = crop_line(image, line_box)
        save_debug_image(os.path.join(roi_dir, "%03d.png" % index), line_image)


def line_box_to_payload(line_box: LineBox) -> Dict[str, int]:
    return {
        "top": int(line_box.top),
        "bottom": int(line_box.bottom),
        "left": int(line_box.left),
        "right": int(line_box.right),
    }


def normalize_line_number(text: str) -> Optional[int]:
    translation = str(text).replace("i", "1").replace("I", "1").replace("l", "1").replace("O", "0").replace("o", "0")
    compact = "".join(ch for ch in translation if ch.isdigit())
    if not compact:
        return None
    try:
        return int(compact)
    except ValueError:
        return None


def repair_line_number_sequence(lines: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    repaired: List[Dict[str, object]] = []
    previous_line_no: Optional[int] = None
    for index, item in enumerate(lines):
        current = dict(item)
        line_no = current.get("line_no")
        if line_no is None:
            if previous_line_no is not None:
                line_no = previous_line_no + 1
            elif index + 1 < len(lines):
                future = next((entry.get("line_no") for entry in lines[index + 1 :] if entry.get("line_no") is not None), None)
                if future is not None:
                    line_no = max(1, int(future) - 1)
        elif previous_line_no is not None and int(line_no) <= previous_line_no:
            line_no = previous_line_no + 1
        current["line_no"] = int(line_no) if line_no is not None else None
        previous_line_no = current["line_no"] if current["line_no"] is not None else previous_line_no
        repaired.append(current)
    return repaired


def pair_line_numbers(number_lines: Sequence[Dict[str, object]], code_lines: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    if not code_lines:
        return []

    if not number_lines:
        return [
            {
                "index": index,
                "line_no": index,
                "text": code_line["text"],
                "code_box": code_line["box"],
                "line_number_box": None,
            }
            for index, code_line in enumerate(code_lines, start=1)
        ]

    available_numbers = list(number_lines)
    structured_lines: List[Dict[str, object]] = []
    for index, code_line in enumerate(code_lines, start=1):
        best_number = None
        best_distance = None
        code_center = float(code_line["box"]["top"] + code_line["box"]["bottom"]) / 2.0
        for candidate in available_numbers:
            candidate_center = float(candidate["box"]["top"] + candidate["box"]["bottom"]) / 2.0
            distance = abs(candidate_center - code_center)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_number = candidate
        line_no = normalize_line_number(str(best_number.get("text", ""))) if best_number is not None else None
        structured_lines.append(
            {
                "index": index,
                "line_no": line_no,
                "text": code_line["text"],
                "code_box": code_line["box"],
                "line_number_box": best_number["box"] if best_number is not None else None,
            }
        )
    return repair_line_number_sequence(structured_lines)


def score_lines(lines: Sequence[Dict[str, object]]) -> Tuple[int, int]:
    non_empty = sum(1 for item in lines if str(item.get("text", "")).strip())
    text_length = sum(len(str(item.get("text", "")).strip()) for item in lines)
    return non_empty, text_length


def recognize_roi_lines(
    recognizer: object,
    crop: np.ndarray,
    roi_name: str,
    segmentation_cfg: Dict[str, object],
    debug_dir: Optional[str],
) -> List[Dict[str, object]]:
    line_boxes = detect_line_boxes(crop, segmentation_cfg, roi_name)
    if debug_dir:
        save_debug_lines(debug_dir, roi_name, crop, line_boxes)
    line_images = [crop_line(crop, line_box) for line_box in line_boxes]
    line_texts = recognizer.recognize_batch(line_images)
    results = []
    for index, (line_box, text) in enumerate(zip(line_boxes, line_texts), start=1):
        results.append(
            {
                "index": index,
                "text": text,
                "box": line_box_to_payload(line_box),
            }
        )
    return results


def recognize_lines_from_boxes(
    recognizer: object,
    crop: np.ndarray,
    line_boxes: Sequence[LineBox],
    debug_dir: Optional[str],
    debug_name: str,
) -> List[Dict[str, object]]:
    if debug_dir:
        save_debug_lines(debug_dir, debug_name, crop, line_boxes)
    line_images = [crop_line(crop, line_box) for line_box in line_boxes]
    line_texts = recognizer.recognize_batch(line_images)
    return [
        {
            "index": index,
            "text": text,
            "box": line_box_to_payload(line_box),
        }
        for index, (line_box, text) in enumerate(zip(line_boxes, line_texts), start=1)
    ]


def recognize_code_lines_with_anchors(
    recognizer: object,
    crop: np.ndarray,
    number_lines: Sequence[Dict[str, object]],
    debug_dir: Optional[str],
) -> List[Dict[str, object]]:
    anchors = line_boxes_from_number_lines(number_lines, crop.shape[1])
    line_boxes = build_code_line_boxes_from_anchors(crop, anchors)
    return recognize_lines_from_boxes(recognizer, crop, line_boxes, debug_dir, "code_anchor")


def recognize_full_roi(recognizer: object, crop: np.ndarray) -> str:
    texts = recognizer.recognize_batch([crop])
    return texts[0] if texts else ""


def prepare_template_roi_context(
    layout_path: str,
    provider_override: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    layout = load_json(layout_path)
    reference_path = resolve_layout_path(layout_path, str(layout["reference_image"]))
    reference = load_image(reference_path)
    recognizer_cfg = dict(layout.get("recognizer", {}))
    if provider_override:
        recognizer_cfg["providers"] = list(provider_override)
    recognizer, recognizer_available, missing_paths = load_recognizer(layout_path, recognizer_cfg)
    providers = list(getattr(recognizer, "providers", ["CPUExecutionProvider"])) if recognizer_available else ["CPUExecutionProvider"]
    segmentation_cfg = dict(layout.get("segmentation", {}))
    rois = build_rois(layout)
    return {
        "layout_path": os.path.abspath(layout_path),
        "layout": layout,
        "reference_path": reference_path,
        "reference": reference,
        "recognizer": recognizer,
        "recognizer_available": recognizer_available,
        "missing_paths": missing_paths,
        "providers": providers,
        "segmentation_cfg": segmentation_cfg,
        "rois": rois,
    }


def run_template_roi_with_context(
    context: Dict[str, object],
    image: np.ndarray,
    image_label: str,
    debug_dir: Optional[str] = None,
) -> Dict[str, object]:
    layout = dict(context["layout"])
    reference = context["reference"]
    recognizer = context["recognizer"]
    recognizer_available = bool(context["recognizer_available"])
    missing_paths = list(context["missing_paths"])
    providers = list(context["providers"])
    segmentation_cfg = dict(context["segmentation_cfg"])
    rois = list(context["rois"])

    aligned = align_to_reference(reference, image, layout)

    roi_results = []
    roi_crops: Dict[str, np.ndarray] = {}
    for roi in rois:
        crop = crop_roi(aligned, roi)
        roi_crops[roi.name] = crop
        if debug_dir:
            save_debug_crop(debug_dir, roi, crop)

        roi_payload = {
            "name": roi.name,
            "box": {"x": roi.x, "y": roi.y, "width": roi.width, "height": roi.height},
        }
        if roi.name in {"line_numbers", "code"}:
            roi_payload["lines"] = recognize_roi_lines(recognizer, crop, roi.name, segmentation_cfg, debug_dir)
        else:
            roi_payload["text"] = recognize_full_roi(recognizer, crop)
        roi_results.append(roi_payload)

    roi_map = {roi.name: roi for roi in rois}
    line_numbers_roi = next((item for item in roi_results if item["name"] == "line_numbers"), {"lines": []})
    code_roi = next((item for item in roi_results if item["name"] == "code"), {"lines": []})
    if (
        line_numbers_roi.get("lines")
        and roi_map.get("code") is not None
        and not uses_fixed_grid(segmentation_cfg, "code")
    ):
        code_crop = roi_crops["code"]
        anchor_boxes = line_boxes_from_number_lines(list(line_numbers_roi.get("lines", [])), code_crop.shape[1])
        if is_reasonable_line_boxes(anchor_boxes, code_crop.shape[0], segmentation_cfg):
            anchored_lines = recognize_code_lines_with_anchors(
                recognizer,
                code_crop,
                list(line_numbers_roi.get("lines", [])),
                debug_dir,
            )
            projected_lines = list(code_roi.get("lines", []))
            if score_lines(anchored_lines) >= score_lines(projected_lines):
                code_roi["lines"] = anchored_lines
    structured_lines = pair_line_numbers(
        list(line_numbers_roi.get("lines", [])),
        list(code_roi.get("lines", [])),
    )

    return {
        "image": image_label,
        "layout": str(context["layout_path"]),
        "reference_image": str(context["reference_path"]),
        "recognizer_available": recognizer_available,
        "missing_model_paths": missing_paths,
        "providers": providers,
        "line_numbers_enabled": any(roi.name == "line_numbers" for roi in rois),
        "rois": roi_results,
        "structured_lines": structured_lines,
    }


def run_template_roi(
    layout_path: str,
    image: np.ndarray,
    image_label: str,
    debug_dir: Optional[str] = None,
    provider_override: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    context = prepare_template_roi_context(layout_path, provider_override=provider_override)
    return run_template_roi_with_context(context, image, image_label, debug_dir=debug_dir)


def main() -> int:
    args = parse_args()
    image = load_image(args.image)
    payload = run_template_roi(
        layout_path=args.layout,
        image=image,
        image_label=os.path.abspath(args.image),
        debug_dir=args.debug_dir,
    )
    write_json(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
