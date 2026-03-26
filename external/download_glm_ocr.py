#!/usr/bin/env python3
from __future__ import annotations

from huggingface_hub import snapshot_download


def main() -> int:
    path = snapshot_download(
        repo_id="zai-org/GLM-OCR",
        local_dir="external_models/GLM-OCR",
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=1,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
