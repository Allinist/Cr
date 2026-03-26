$ErrorActionPreference = "Stop"

$repo = "C:\Users\Tom\Documents\Projects\CodeReader"
Set-Location $repo

$env:PYTHONPATH = (Resolve-Path ".\.venv314\Lib\site-packages").Path
$python = "C:\Users\Tom\AppData\Local\Python\pythoncore-3.14-64\python.exe"

& $python ".\external\obs_roi_sync.py" `
  --layout ".\config\template_roi_layout.example.json" `
  --ocr-output-dir ".\external_obs_ocr_roi_only" `
  --code-output-dir ".\external_obs_code_roi_tmp" `
  --capture-dir ".\external_obs_captures" `
  --projection-config ".\config\projection.json" `
  --skip-capture `
  --skip-alignment `
  --session-name "obs_roi_roi_only" `
  --ocr-accel "auto"

if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

& $python ".\external\build_raw_ocr_code.py" `
  --ocr-root ".\external_obs_ocr_roi_only" `
  --output-root ".\external_obs_code_raw_roi_only"

if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

& $python ".\external\build_fused_ocr_code.py" `
  --ocr-root ".\external_obs_ocr_roi_only" `
  --output-root ".\external_obs_code_fused_roi_only"

if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

& $python ".\external\build_capture_completeness_report.py" `
  --summary ".\external_obs_ocr_roi_only\obs_roi_roi_only.summary.json" `
  --output ".\external_obs_ocr_roi_only\obs_roi_roi_only.capture_report.json"

exit $LASTEXITCODE
