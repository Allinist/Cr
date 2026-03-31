#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from typing import Dict, List

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe GLM-OCR ONNX export feasibility on a local sample image.")
    parser.add_argument("--model-path", required=True, help="Local GLM-OCR model path.")
    parser.add_argument("--image", required=True, help="Sample image path.")
    parser.add_argument("--output-dir", required=True, help="Directory for ONNX probe artifacts.")
    return parser.parse_args()


class VisionWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, image_grid_thw):
        output = self.model.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=True,
        ).pooler_output
        return torch.cat(output, dim=0)


class DecoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.text = model.model.language_model
        self.lm_head = model.lm_head

    def forward(self, inputs_embeds, attention_mask, position_ids):
        output = self.text(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        return self.lm_head(output[0])


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_inputs(model_path: str, image_path: str):
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=False,
    )
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Text Recognition:"},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    return processor, inputs


def export_probe(model_path: str, image_path: str, output_dir: str, attn_mode: str, component: str) -> Dict[str, object]:
    started = time.perf_counter()
    report: Dict[str, object] = {
        "component": component,
        "attn_mode": attn_mode,
        "status": "error",
    }
    kwargs = {
        "pretrained_model_name_or_path": model_path,
        "local_files_only": True,
        "trust_remote_code": True,
    }
    if attn_mode == "eager":
        kwargs["attn_implementation"] = "eager"
    model = AutoModelForImageTextToText.from_pretrained(**kwargs)
    model.eval()
    _, inputs = build_inputs(model_path, image_path)
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, "glm_ocr_%s_%s.onnx" % (component, attn_mode))
    try:
        if component == "vision":
            wrapper = VisionWrapper(model)
            export_args = (inputs["pixel_values"], inputs["image_grid_thw"])
            torch.onnx.export(
                wrapper,
                export_args,
                output_path,
                input_names=["pixel_values", "image_grid_thw"],
                output_names=["image_embeds"],
                dynamic_axes={
                    "pixel_values": {0: "patch_tokens"},
                    "image_grid_thw": {0: "num_images"},
                    "image_embeds": {0: "vision_tokens"},
                },
                opset_version=17,
                do_constant_folding=True,
            )
        elif component == "decoder":
            inputs_embeds = model.model.get_input_embeddings()(inputs["input_ids"])
            image_embeds = model.get_image_features(
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
                return_dict=True,
            ).pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = model.model.get_placeholder_mask(
                inputs["input_ids"], inputs_embeds, image_features=image_embeds
            )
            merged_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            position_ids = model.model.compute_3d_position_ids(
                input_ids=inputs["input_ids"],
                image_grid_thw=inputs["image_grid_thw"],
                inputs_embeds=merged_embeds,
                attention_mask=inputs["attention_mask"],
                mm_token_type_ids=inputs["mm_token_type_ids"],
            )
            wrapper = DecoderWrapper(model)
            export_args = (merged_embeds, inputs["attention_mask"], position_ids)
            torch.onnx.export(
                wrapper,
                export_args,
                output_path,
                input_names=["inputs_embeds", "attention_mask", "position_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "inputs_embeds": {1: "sequence"},
                    "attention_mask": {1: "sequence"},
                    "position_ids": {2: "sequence"},
                    "logits": {1: "sequence"},
                },
                opset_version=17,
                do_constant_folding=True,
            )
        else:
            raise RuntimeError("unsupported component: %s" % component)
        report["status"] = "ok"
        report["output_path"] = output_path
        report["output_size"] = os.path.getsize(output_path)
    except Exception as exc:
        report["error"] = str(exc).strip() or ("%s: %r" % (type(exc).__name__, exc))
        report["traceback"] = traceback.format_exc()
    report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    return report


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_dir)
    results: List[Dict[str, object]] = []
    for attn_mode in ("default", "eager"):
        for component in ("vision", "decoder"):
            results.append(export_probe(args.model_path, args.image, args.output_dir, attn_mode, component))
    report_path = os.path.join(args.output_dir, "glm_ocr_onnx_probe_report.json")
    with open(report_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
