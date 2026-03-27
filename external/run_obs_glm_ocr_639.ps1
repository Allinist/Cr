$ErrorActionPreference = "Stop"
Set-Location "C:\Users\Tom\Documents\Projects\CodeReader"
$env:PYTHONPATH = (Resolve-Path ".\tools\python310_embed\Lib\site-packages").Path
$python = (Resolve-Path ".\tools\python310_embed\python.exe").Path
& $python ".\external\obs_glm_ocr_sync.py" `
  --skip-capture `
  --layout ".\config\template_roi_layout.glm_ocr.example.json" `
  --capture-dir ".\external_obs_captures" `
  --ocr-output-dir ".\external_out_glm_obs\ocr_639" `
  --code-output-dir ".\external_out_glm_obs\code_639" `
  --session-name "obs_glm_roi_639" `
  --glm-model-path ".\external_models\GLM-OCR" `
  --glm-device "directml" `
  --glm-target "all" `
  --glm-line-mode "segmented" `
  --glm-max-new-tokens 512 `
  --glm-local-files-only `
  --emit-partial
exit $LASTEXITCODE
