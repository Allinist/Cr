#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List

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

        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=local_files_only, trust_remote_code=True)
        self.device, self.device_label = resolve_device(device_name)
        torch_dtype = "auto"
        if self.device_label == "directml":
            # DirectML does not support bfloat16 for this model path, so load in float32.
            torch_dtype = torch.float32
        kwargs = {
            "pretrained_model_name_or_path": model_path,
            "torch_dtype": torch_dtype,
            "local_files_only": local_files_only,
            "trust_remote_code": True,
        }

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


def segment_code_lines(code_crop: np.ndarray, layout: Dict[str, object]) -> List[Dict[str, object]]:
    segmentation_cfg = dict(layout.get("segmentation", {}))
    line_boxes = build_fixed_grid_line_boxes(code_crop, segmentation_cfg, "code")
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
                "image": crop_line(code_crop, line_box),
            }
        )
    return lines


def should_process(target: str, roi_name: str) -> bool:
    if target == "all":
        return roi_name in {"header", "code", "footer"}
    return target == roi_name


def main() -> int:
    args = parse_args()
    layout = load_json(args.layout)
    image = load_image(args.image)
    rois = build_rois(layout)

    recognizer = GlmOcrRecognizer(
        model_path=args.model_path,
        device_name=args.device,
        local_files_only=bool(args.local_files_only),
    )

    roi_payloads: List[Dict[str, object]] = []
    for roi in rois:
        if not should_process(args.target, roi.name):
            continue
        crop = crop_roi(image, roi)
        payload: Dict[str, object] = {
            "name": roi.name,
            "box": {"x": roi.x, "y": roi.y, "width": roi.width, "height": roi.height},
        }
        if roi.name == "code" and args.line_mode == "segmented":
            line_payloads = []
            for entry in segment_code_lines(crop, layout):
                text = recognizer.recognize(entry["image"], "Text Recognition:", args.max_new_tokens)
                line_payloads.append(
                    {
                        "index": entry["index"],
                        "box": entry["box"],
                        "text": text,
                    }
                )
            payload["lines"] = line_payloads
        else:
            payload["text"] = recognizer.recognize(crop, "Text Recognition:", args.max_new_tokens)
        roi_payloads.append(payload)

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
            "rois": roi_payloads,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
