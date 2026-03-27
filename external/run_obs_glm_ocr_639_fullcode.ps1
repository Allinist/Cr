$ErrorActionPreference = "Stop"
Set-Location "C:\Users\Tom\Documents\Projects\CodeReader"
$env:PYTHONPATH = (Resolve-Path ".\tools\python310_embed\Lib\site-packages").Path
$python = (Resolve-Path ".\tools\python310_embed\python.exe").Path
& $python ".\external\obs_glm_ocr_sync.py" `
  --config ".\config\obs_glm_ocr_639_fullcode.json"
exit $LASTEXITCODE
