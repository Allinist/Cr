$ErrorActionPreference = "Stop"

$repo = "C:\Users\Tom\Documents\Projects\CodeReader"
Set-Location $repo

$env:PYTHONPATH = (Resolve-Path ".\tools\python310_embed\Lib\site-packages").Path
$python = (Resolve-Path ".\tools\python310_embed\python.exe").Path

& $python ".\external\glm_ocr_local_runner.py" `
  --image ".\screenshots\test3.png" `
  --layout ".\config\template_roi_layout.glm_ocr.example.json" `
  --output ".\external_out_glm_ocr_local\roi_result.json" `
  --model-path ".\external_models\GLM-OCR" `
  --local-files-only `
  --device "cpu" `
  --target "all" `
  --line-mode "segmented"

exit $LASTEXITCODE
