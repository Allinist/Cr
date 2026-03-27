$ErrorActionPreference = "Stop"
Set-Location "C:\Users\Tom\Documents\Projects\CodeReader"
$env:PYTHONPATH = (Resolve-Path ".\tools\python310_embed\Lib\site-packages").Path
$python = (Resolve-Path ".\tools\python310_embed\python.exe").Path
$supplementDir = "C:\Users\Tom\Documents\Projects\CodeReader\external_obs_captures_incremental"

if (!(Test-Path $supplementDir)) {
  New-Item -ItemType Directory -Path $supplementDir | Out-Null
}

& $python ".\external\obs_glm_ocr_sync.py" `
  --config ".\config\obs_glm_ocr_639_fullcode_supplement.json"

exit $LASTEXITCODE
