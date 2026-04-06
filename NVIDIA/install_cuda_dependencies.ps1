$ErrorActionPreference = "Stop"

if (-not $env:PYTHON_EXECUTABLE) {
  $repo = Split-Path -Parent $PSScriptRoot
  $embeddedPython = Join-Path $repo "NVIDIA\python311_embed\python.exe"
  if (Test-Path $embeddedPython) {
    $env:PYTHON_EXECUTABLE = $embeddedPython
  }
  else {
    $env:PYTHON_EXECUTABLE = "python"
  }
}

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

& $env:PYTHON_EXECUTABLE -m pip install --upgrade pip
& $env:PYTHON_EXECUTABLE -m pip install --upgrade setuptools wheel
& $env:PYTHON_EXECUTABLE -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
& $env:PYTHON_EXECUTABLE -m pip install -r ".\requirements-ocr.txt"
& $env:PYTHON_EXECUTABLE -m pip install -r ".\NVIDIA\requirements-nvidia.txt"
& $env:PYTHON_EXECUTABLE -c "import torch; print('python=', __import__('sys').executable); print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('cuda_device_count=', torch.cuda.device_count())"

exit $LASTEXITCODE
