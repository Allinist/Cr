#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local-only GLM-OCR experiment runner for ROI crops.")
    parser.add_argument("--image", required=True, help="Input screenshot path.")
    parser.add_argument("--layout", required=True, help="ROI layout JSON path.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument(
        "--model-path",
        default="zai-org/GLM-OCR",
        help="Local model directory or HF model id. Use --local-files-only to forbid remote fetch.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "directml"],
        default="auto",
        help="Preferred local inference device.",
    )
    parser.add_argument(
        "--target",
        choices=["code", "header", "footer", "all"],
        default="all",
        help="Which ROI content to run through GLM-OCR.",
    )
    parser.add_argument(
        "--line-mode",
        choices=["full", "segmented"],
        default="segmented",
        help="For code ROI, run once on the whole block or once per segmented line.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load the model from local cache or a local directory.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation length.")
    parser.add_argument(
        "--full-max-edge",
        type=int,
        default=0,
        help="When line-mode=full, downscale the code ROI so its longest edge is at most this value. 0 disables.",
    )
    return parser.parse_args()


def load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)


def write_json(path: str, payload: Dict[str, object]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def pil_image_from_bgr(image: np.ndarray):
    from PIL import Image

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def resolve_device(preferred: str):
    import torch

    normalized = preferred.lower()
    if normalized == "cpu":
        return torch.device("cpu"), "cpu"
    if normalized == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        raise RuntimeError("CUDA requested but torch.cuda is not available.")
    if normalized == "directml":
        try:
            import torch_directml

            return torch_directml.device(), "directml"
        except Exception as exc:  # pragma: no cover - depends on local env
            raise RuntimeError("DirectML requested but torch-directml is not available.") from exc

    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        try:
            import torch_directml

            return torch_directml.device(), "directml"
        except Exception:
            return torch.device("cpu"), "cpu"

    return torch.device("cpu"), "cpu"


class GlmOcrRecognizer:
    def __init__(self, model_path: str, device_name: str, local_files_only: bool):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            trust_remote_code=True,
            use_fast=False,
        )
        self.device, self.device_label = resolve_device(device_name)
        model_dtype = "auto"
        if self.device_label == "directml":
            # DirectML does not support bfloat16 for this model path, so load in float32.
            model_dtype = torch.float32
        kwargs = {
            "pretrained_model_name_or_path": model_path,
            "dtype": model_dtype,
            "local_files_only": local_files_only,
            "trust_remote_code": True,
        }
        if self.device_label == "directml":
            # DirectML currently has runtime issues on larger GLM-OCR attention shapes with SDPA.
            kwargs["attn_implementation"] = "eager"

        if self.device_label == "cuda":
            kwargs["device_map"] = "auto"
            self.model = AutoModelForImageTextToText.from_pretrained(**kwargs)
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(**kwargs)
            self.model.to(self.device)
        self.model.eval()
        self.torch = torch

    def recognize(self, image: np.ndarray, prompt: str, max_new_tokens: int) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image_from_bgr(image)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        inputs.pop("token_type_ids", None)
        with self.torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        prompt_length = inputs["input_ids"].shape[1]
        return self.processor.decode(generated_ids[0][prompt_length:], skip_special_tokens=True).strip()


def crop_roi(image: np.ndarray, roi: Roi) -> np.ndarray:
    x1 = max(0, roi.x)
    y1 = max(0, roi.y)
    x2 = min(image.shape[1], roi.x + roi.width)
    y2 = min(image.shape[0], roi.y + roi.height)
    return image[y1:y2, x1:x2]


def crop_line(image: np.ndarray, line_box: LineBox) -> np.ndarray:
    return image[line_box.top:line_box.bottom, line_box.left:line_box.right]


def maybe_resize_long_edge(image: np.ndarray, max_edge: int) -> np.ndarray:
    if max_edge <= 0:
        return image
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_edge:
        return image
    scale = float(max_edge) / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def build_fixed_grid_line_boxes(image: np.ndarray, segmentation_cfg: Dict[str, object], roi_name: str) -> List[LineBox]:
    line_count_key = "%s_expected_line_count" % roi_name
    expected_line_count = int(segmentation_cfg.get(line_count_key, segmentation_cfg.get("expected_line_count", 0)))
    if expected_line_count <= 0:
        return []
    top_inset = int(segmentation_cfg.get("%s_top_inset" % roi_name, 0))
    bottom_inset = int(segmentation_cfg.get("%s_bottom_inset" % roi_name, 0))
    left_inset = int(segmentation_cfg.get("%s_left_inset" % roi_name, 0))
    right_inset = int(segmentation_cfg.get("%s_right_inset" % roi_name, 0))
    start_shift_px = float(segmentation_cfg.get("%s_start_shift_px" % roi_name, 0.0))
    line_height_scale = float(segmentation_cfg.get("%s_line_height_scale" % roi_name, 1.0))
    line_height_shrink_px = float(segmentation_cfg.get("%s_line_height_shrink_px" % roi_name, 0.0))
    center_squeeze_px = float(segmentation_cfg.get("%s_center_squeeze_px" % roi_name, 0.0))

    usable_top = max(0, top_inset)
    usable_bottom = max(usable_top + 1, image.shape[0] - max(0, bottom_inset))
    usable_left = max(0, left_inset)
    usable_right = max(usable_left + 1, image.shape[1] - max(0, right_inset))
    usable_height = usable_bottom - usable_top
    line_height = float(usable_height) / float(expected_line_count)
    scaled_height = max(1.0, line_height * max(0.5, min(1.5, line_height_scale)) - max(0.0, line_height_shrink_px))
    center_index = (expected_line_count - 1) / 2.0

    boxes: List[LineBox] = []
    for index in range(expected_line_count):
        base_center = usable_top + (index + 0.5) * line_height + start_shift_px
        normalized = ((index - center_index) / center_index) if center_index > 0 else 0.0
        center = base_center - normalized * center_squeeze_px
        top = int(round(center - scaled_height / 2.0))
        bottom = int(round(center + scaled_height / 2.0))
        top = max(usable_top, top)
        bottom = min(usable_bottom, max(top + 1, bottom))
        boxes.append(LineBox(top=top, bottom=bottom, left=usable_left, right=usable_right))
    return boxes


def segment_roi_lines(roi_crop: np.ndarray, layout: Dict[str, object], roi_name: str) -> List[Dict[str, object]]:
    segmentation_cfg = dict(layout.get("segmentation", {}))
    line_boxes = build_fixed_grid_line_boxes(roi_crop, segmentation_cfg, roi_name)
    lines: List[Dict[str, object]] = []
    for index, line_box in enumerate(line_boxes, start=1):
        lines.append(
            {
                "index": index,
                "box": {
                    "top": int(line_box.top),
                    "bottom": int(line_box.bottom),
                    "left": int(line_box.left),
                    "right": int(line_box.right),
                },
                "image": crop_line(roi_crop, line_box),
            }
        )
    return lines


def segment_code_lines(code_crop: np.ndarray, layout: Dict[str, object]) -> List[Dict[str, object]]:
    return segment_roi_lines(code_crop, layout, "code")


def line_numbers_enabled(layout: Dict[str, object]) -> bool:
    cfg = layout.get("line_numbers", {})
    if not isinstance(cfg, dict):
        return False
    return bool(cfg.get("enabled", any(str(item.get("name")) == "line_numbers" for item in layout.get("rois", []))))


def normalize_visual_line_number(text: str) -> Tuple[Optional[int], bool]:
    value = str(text).strip().replace(" ", "")
    if not value:
        return None, False
    continued = "+" in value
    value = (
        value.replace("I", "1")
        .replace("l", "1")
        .replace("|", "1")
        .replace("O", "0")
        .replace("o", "0")
    )
    digits = "".join(ch for ch in value if ch.isdigit())
    if not digits:
        return None, continued
    return int(digits), continued


def build_structured_lines(
    code_lines: List[Dict[str, object]],
    number_lines: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    structured: List[Dict[str, object]] = []
    last_line_no: Optional[int] = None
    for index, code_line in enumerate(code_lines, start=1):
        number_line = number_lines[index - 1] if index - 1 < len(number_lines) else {}
        number_text = str(number_line.get("text", "")).strip()
        parsed_line_no, is_continuation = normalize_visual_line_number(number_text)
        effective_line_no = parsed_line_no if parsed_line_no is not None else last_line_no
        if effective_line_no is None:
            effective_line_no = index
        if parsed_line_no is not None:
            last_line_no = parsed_line_no
        structured.append(
            {
                "index": index,
                "line_no": effective_line_no,
                "line_no_text": number_text,
                "continued": bool(is_continuation),
                "text": str(code_line.get("text", "")),
                "code_box": code_line.get("box"),
                "number_box": number_line.get("box"),
            }
        )
    return structured


def should_process(target: str, roi_name: str) -> bool:
    if target == "all":
        return roi_name in {"header", "line_numbers", "code", "footer"}
    return target == roi_name


def main() -> int:
    args = parse_args()
    layout = load_json(args.layout)
    image = load_image(args.image)
    rois = build_rois(layout)
    use_line_numbers = line_numbers_enabled(layout)

    recognizer = GlmOcrRecognizer(
        model_path=args.model_path,
        device_name=args.device,
        local_files_only=bool(args.local_files_only),
    )

    roi_payloads: List[Dict[str, object]] = []
    code_line_payloads: List[Dict[str, object]] = []
    line_number_payloads: List[Dict[str, object]] = []
    for roi in rois:
        if roi.name == "line_numbers" and not use_line_numbers:
            continue
        if not should_process(args.target, roi.name):
            continue
        crop = crop_roi(image, roi)
        payload: Dict[str, object] = {
            "name": roi.name,
            "box": {"x": roi.x, "y": roi.y, "width": roi.width, "height": roi.height},
        }
        if roi.name in {"code", "line_numbers"} and args.line_mode == "segmented":
            line_payloads = []
            for entry in segment_roi_lines(crop, layout, roi.name):
                text = recognizer.recognize(entry["image"], "Text Recognition:", args.max_new_tokens)
                line_payloads.append(
                    {
                        "index": entry["index"],
                        "box": entry["box"],
                        "text": text,
                    }
                )
            payload["lines"] = line_payloads
            if roi.name == "code":
                code_line_payloads = line_payloads
            elif roi.name == "line_numbers":
                line_number_payloads = line_payloads
        else:
            if roi.name == "code" and args.line_mode == "full":
                crop = maybe_resize_long_edge(crop, int(args.full_max_edge))
            payload["text"] = recognizer.recognize(crop, "Text Recognition:", args.max_new_tokens)
        roi_payloads.append(payload)

    structured_lines: List[Dict[str, object]] = []
    if args.line_mode == "segmented" and code_line_payloads:
        structured_lines = (
            build_structured_lines(code_line_payloads, line_number_payloads)
            if use_line_numbers and line_number_payloads
            else [
                {
                    "index": item["index"],
                    "line_no": item["index"],
                    "line_no_text": str(item["index"]),
                    "continued": False,
                    "text": str(item["text"]),
                    "code_box": item.get("box"),
                    "number_box": None,
                }
                for item in code_line_payloads
            ]
        )

    write_json(
        args.output,
        {
            "image": os.path.abspath(args.image),
            "layout": os.path.abspath(args.layout),
            "model_path": args.model_path,
            "local_files_only": bool(args.local_files_only),
            "device": recognizer.device_label,
            "target": args.target,
            "line_mode": args.line_mode,
            "line_numbers_enabled": use_line_numbers,
            "rois": roi_payloads,
            "structured_lines": structured_lines,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
