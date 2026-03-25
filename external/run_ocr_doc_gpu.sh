#!/usr/bin/env bash
set -euo pipefail

IMAGE_PATH="${1:-./screenshots/test.png}"
WORKSPACE_PATH="${2:-./external_out_doc}"

python external/sync_manager.py \
  --image "${IMAGE_PATH}" \
  --workspace "${WORKSPACE_PATH}" \
  --ocr-mode balanced \
  --accel auto \
  --model-preset doc
