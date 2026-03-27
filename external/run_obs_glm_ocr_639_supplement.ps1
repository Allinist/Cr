$ErrorActionPreference = "Stop"
Set-Location "C:\Users\Tom\Documents\Projects\CodeReader"
$env:PYTHONPATH = (Resolve-Path ".\tools\python310_embed\Lib\site-packages").Path
$python = (Resolve-Path ".\tools\python310_embed\python.exe").Path
$supplementDir = "C:\Users\Tom\Documents\Projects\CodeReader\external_obs_captures_incremental"

if (!(Test-Path $supplementDir)) {
  New-Item -ItemType Directory -Path $supplementDir | Out-Null
}

& $python ".\external\obs_glm_ocr_sync.py" `
  --skip-capture `
  --layout ".\config\template_roi_layout.glm_ocr.example.json" `
  --capture-dir $supplementDir `
  --ocr-output-dir ".\external_out_glm_obs\ocr_639_incremental" `
  --code-output-dir ".\external_out_glm_obs\code_639_incremental" `
  --session-name "obs_glm_roi_639_incremental" `
  --glm-model-path ".\external_models\GLM-OCR" `
  --glm-device "directml" `
  --glm-target "all" `
  --glm-line-mode "segmented" `
  --glm-max-new-tokens 512 `
  --glm-local-files-only `
  --emit-partial

exit $LASTEXITCODE
