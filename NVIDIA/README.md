# NVIDIA CUDA Batch Version

这个目录提供一个不改动原有 `external/` 代码的 CUDA 批处理版本。

## 目录说明

- `batch_glm_ocr_cuda.py`
  - 顺序读取图片文件夹中的图片
  - 调用现有 `GLM-OCR + ROI` 识别逻辑
  - 复用现有分页合并与代码重建逻辑
- `run_glm_ocr_cuda_folder.ps1`
  - Windows PowerShell 启动脚本
- `install_cuda_dependencies.ps1`
  - CUDA 环境依赖安装脚本
- `install_portable_python.ps1`
  - 在项目内安装独立便携 Python 3.11
- `download_glm_ocr_model.ps1`
  - 使用 Hugging Face Token 下载 `GLM-OCR`
- `config/glm_ocr_cuda_folder.example.json`
  - CUDA 版批处理配置样例
- `requirements-nvidia.txt`
  - 额外 Python 依赖清单

## 建议环境

建议直接使用项目内的便携 Python，这样不会影响系统里的其他 Python 环境：

```powershell
.\NVIDIA\install_portable_python.ps1
.\NVIDIA\install_cuda_dependencies.ps1
```

`install_portable_python.ps1` 默认下载 Python 3.11.9 的 Windows 64-bit embeddable package。

Python 下载源来自 Python 官方页面：

- [Python 3.11.9 release page](https://www.python.org/downloads/release/python-3119/)
- [Python Windows downloads](https://www.python.org/downloads/windows/)

如果你的 CUDA 版本不是 `cu124`，把 `install_cuda_dependencies.ps1` 里的 PyTorch 安装源改成你机器实际可用的版本。

## 运行方式

当前示例配置已经把图片目录设置成：

- `E:\Personal\1spc-modules`
- 默认 layout 已切到 `NVIDIA/config/template_roi_layout.glm_ocr.1920x1080.sample.json`

正式运行前建议确认：

- `NVIDIA/config/glm_ocr_cuda_folder.example.json`
  - `layout`: 你的 ROI 布局文件
  - `glm_model_path`: 你的本地 GLM-OCR 模型目录

模型下载命令我也单独做成了脚本。你把 Hugging Face Token 放进环境变量后执行：

```powershell
$env:HF_TOKEN = "hf_xxx"
.\NVIDIA\download_glm_ocr_model.ps1
```

如果你想手动执行命令，等便携 Python 和依赖装好后可以直接运行：

```powershell
$env:HF_TOKEN = "hf_xxx"
.\NVIDIA\python311_embed\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='zai-org/GLM-OCR', local_dir=r'.\external_models\GLM-OCR', token=r'$env:HF_TOKEN')"
```

然后执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\NVIDIA\run_glm_ocr_cuda_folder.ps1
```

如果你想直接传参运行：

```powershell
python .\NVIDIA\batch_glm_ocr_cuda.py `
  --input-dir .\screenshots `
  --layout .\config\template_roi_layout.glm_ocr.example.json `
  --ocr-output-dir .\NVIDIA\out\ocr `
  --code-output-dir .\NVIDIA\out\code `
  --glm-model-path .\external_models\GLM-OCR `
  --glm-device cuda `
  --glm-line-mode segmented `
  --glm-local-files-only
```

## 输出内容

- `NVIDIA/out/ocr/<session>.images/`
  - 每张图片的 ROI 识别结果
- `NVIDIA/out/ocr/*.ocr.json`
  - 同一文件多页合并后的 OCR 结果
- `NVIDIA/out/code/`
  - 最终重建代码
- `NVIDIA/out/ocr/<session>.state.json`
  - 会话状态，可用于断点续跑
- `NVIDIA/out/ocr/<session>.summary.json`
  - 本次批处理摘要

## 说明

- 图片按文件名自然顺序处理，例如 `1.png, 2.png, 10.png`。
- 默认会跳过已经成功处理过的图片；可用 `--no-skip-existing` 关闭。
- 如果截图头部仍然包含 `FILE / PAGE / LINES` 信息，就会自动按页合并并重建完整文件。
- 你当前这批图片实测是 `1920x1080`，所以我额外提供了一份按这个分辨率缩放过的 ROI 布局样例。
