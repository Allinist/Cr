# CodeReader

This workspace contains the first-pass implementation skeleton for the intranet
code projection flow described in
[`INTRANET_OCR_SYNC_DESIGN.md`](C:\Users\Tom\Documents\Projects\CodeReader\INTRANET_OCR_SYNC_DESIGN.md).

## Current scope

- `intranet/build_manifest.py`: scan files and generate stable page/slice metadata
- `intranet/render_pages.py`: render pages with `tkinter` or stdout fallback
- `intranet/list_files.sh`: simple file discovery helper
- `intranet/compute_diff.sh`: thin wrapper around `git diff --name-status`
- `intranet/run_sync.sh`: build and render in one step
- `external/obs_capture.py`: capture screenshots from OBS websocket
- `external/page_detector.py`: parse projector page headers
- `external/ocr_runner.py`: run PaddleOCR on screenshots
- `external/code_rebuilder.py`: parse line-numbered OCR output into a structured form
- `external/sync_manager.py`: orchestrate the first-pass external OCR flow

## Supported file handling

- Text OCR flow: `.java`, `.xml`, `.sql`, `.properties`, `.yml`, `.yaml`, `.sh`, `.md`, `.txt`
- Binary index only: `.class`, `.jar`

## Python environment

Install the OCR-side dependencies inside your virtual environment with:

```bash
pip install -r requirements-ocr.txt
```

## Example

```bash
python3 intranet/build_manifest.py \
  --repo-root /repo/spc-cloud \
  --output /tmp/manifest.json \
  --branch feature/demo \
  --commit abc1234

python3 intranet/render_pages.py \
  --manifest /tmp/manifest.json \
  --renderer stdout \
  --limit 3 \
  --dwell-ms 100

python external/ocr_runner.py \
  --image capture.png \
  --output external_out/ocr.json

python external/sync_manager.py \
  --image capture.png \
  --workspace external_out
```

## Notes

- The manifest keeps full page bodies for now so the first version stays simple.
- Diff-aware page selection is still the next major step.
- The external OCR flow is now scaffolded, but language-aware repair and OBS authentication are still minimal.
