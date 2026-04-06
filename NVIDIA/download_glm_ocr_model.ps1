$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

if (-not $env:PYTHON_EXECUTABLE) {
  $embeddedPython = Join-Path $repo "NVIDIA\python311_embed\python.exe"
  if (Test-Path $embeddedPython) {
    $env:PYTHON_EXECUTABLE = $embeddedPython
  }
  else {
    $env:PYTHON_EXECUTABLE = "python"
  }
}

if (-not $env:HF_TOKEN) {
  throw "HF_TOKEN is not set. Example: `$env:HF_TOKEN = 'hf_xxx'"
}

$targetDir = ".\external_models\GLM-OCR"
New-Item -ItemType Directory -Force -Path ".\external_models" | Out-Null

& $env:PYTHON_EXECUTABLE -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='zai-org/GLM-OCR', local_dir=r'.\\external_models\\GLM-OCR', token=r'$env:HF_TOKEN')"

exit $LASTEXITCODE
