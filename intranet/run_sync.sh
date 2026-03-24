#!/usr/bin/env bash
set -eu

REPO_ROOT="${1:-.}"
OUTPUT_DIR="${2:-./out}"
BRANCH="${BRANCH:-unknown}"
COMMIT="${COMMIT:-unknown}"
PAGE_LINES="${PAGE_LINES:-40}"
PAGE_COLS="${PAGE_COLS:-110}"
DWELL_MS="${DWELL_MS:-1800}"

mkdir -p "$OUTPUT_DIR"

python3 intranet/build_manifest.py \
  --repo-root "$REPO_ROOT" \
  --output "$OUTPUT_DIR/manifest.json" \
  --branch "$BRANCH" \
  --commit "$COMMIT" \
  --page-lines "$PAGE_LINES" \
  --page-cols "$PAGE_COLS"

python3 intranet/render_pages.py \
  --manifest "$OUTPUT_DIR/manifest.json" \
  --dwell-ms "$DWELL_MS"
