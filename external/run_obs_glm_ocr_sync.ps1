$ErrorActionPreference = "Stop"

$repo = "C:\Users\Tom\Documents\Projects\CodeReader"
Set-Location $repo

$env:PYTHONPATH = (Resolve-Path ".\tools\python310_embed\Lib\site-packages").Path
$python = (Resolve-Path ".\tools\python310_embed\python.exe").Path

$configPath = ".\config\obs_glm_ocr_pipeline.json"
if ($args.Length -gt 0 -and $args[0]) {
  $configPath = $args[0]
}

& $python ".\external\obs_glm_ocr_sync.py" --config $configPath

exit $LASTEXITCODE
