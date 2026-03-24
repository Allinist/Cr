# CodeReader 使用说明

## 1. 目标

这份文档用于说明当前这套工具在：

- 内网 Linux 设备
- 外网 Windows 设备

上分别如何使用。

当前代码已经具备两条主链路的第一版骨架：

- 内网侧：扫描项目、生成分页清单、投屏展示代码页
- 外网侧：接收截图、OCR 识别、解析页面元数据、生成初步重建结果

## 2. 当前目录结构

主要目录和文件如下：

```text
CodeReader/
  config/
    projection.example.json
    projects.example.json
  intranet/
    project_registry.py
    scan_project_tree.py
    build_manifest.py
    render_pages.py
    list_files.sh
    compute_diff.sh
    run_sync.sh
  external/
    obs_capture.py
    page_detector.py
    ocr_runner.py
    code_rebuilder.py
    verifier.py
    sync_manager.py
  requirements-ocr.txt
  README.md
  INTRANET_OCR_SYNC_DESIGN.md
```

## 3. 内网环境怎么用

### 3.1 内网环境要求

建议环境：

- Linux
- Python 3.7.x
- `bash`
- 不依赖额外第三方 Python 包

当前内网脚本都只使用 Python 标准库，目标就是尽量适应老环境。

### 3.2 内网侧解决什么问题

内网侧负责：

- 注册项目根目录
- 选择本次扫描的模块目录
- 扫描目录结构和文件类型
- 生成代码分页与横向切片 manifest
- 将代码按固定格式显示出来，供外网截图 OCR

内网侧不负责 OCR。

### 3.3 配置多个项目

先准备项目配置文件。

将：

- `config/projects.example.json`

复制成：

- `config/projects.json`

然后填写你的项目根目录。例如：

```json
{
  "projects": {
    "spc-cloud": {
      "repo_name": "spc-cloud",
      "project_root": "/data/repos/spc-cloud",
      "scan_root": "/data/repos/spc-cloud",
      "exclude_dirs": [".git", "target", "build", "dist", "node_modules"]
    },
    "spc-order": {
      "repo_name": "spc-order",
      "project_root": "/data/repos/spc-order",
      "scan_root": "/data/repos/spc-order",
      "exclude_dirs": [".git", "target", "build", "dist", "node_modules"]
    }
  }
}
```

说明：

- `project_root` 是稳定项目根目录
- `scan_root` 是默认扫描目录
- 每次执行时仍可临时覆盖 `scan_root`

### 3.4 扫描某个项目的目录结构和文件类型

如果你想先看一个项目下有哪些目录、有哪些文件类型：

```bash
python3 intranet/scan_project_tree.py \
  --project spc-cloud \
  --config config/projects.json \
  --output /tmp/project_scan.json
```

如果你只想扫某个模块：

```bash
python3 intranet/scan_project_tree.py \
  --project spc-cloud \
  --config config/projects.json \
  --scan-root /data/repos/spc-cloud/spc-modules-system \
  --output /tmp/project_scan.json
```

输出结果说明：

- `file_types`: 全局文件类型统计
- `directories`: 每个目录的文件数、子目录数、文件类型分布
- `files`: 每个文件的相对路径、大小、类型

### 3.5 生成用于投屏的 manifest

如果你要把代码分页后交给外网 OCR：

```bash
python3 intranet/build_manifest.py \
  --project spc-cloud \
  --config config/projects.json \
  --scan-root /data/repos/spc-cloud/spc-modules-system \
  --output /tmp/manifest.json \
  --branch feature/demo \
  --commit abc1234
```

常用参数：

- `--page-lines 40`
- `--page-cols 110`
- `--mode full`

示例：

```bash
python3 intranet/build_manifest.py \
  --project spc-cloud \
  --config config/projects.json \
  --scan-root /data/repos/spc-cloud/spc-modules-system \
  --output /tmp/manifest.json \
  --branch feature/demo \
  --commit abc1234 \
  --page-lines 40 \
  --page-cols 110 \
  --mode full
```

manifest 的作用：

- 记录当前扫描的是哪个项目、哪个模块
- 记录每个文件被拆成了多少页、多少横向 slice
- 为每一页生成稳定的 `CHUNK`
- 保证外网 OCR 时有明确的文件路径、行区间、列区间

### 3.6 在内网设备上投屏显示

生成 manifest 后，执行：

```bash
python3 intranet/render_pages.py \
  --manifest /tmp/manifest.json \
  --dwell-ms 1800
```

如果图形界面可用，脚本会优先尝试 `tkinter` 全屏展示。

如果图形界面不可用，可强制 stdout 方式预览：

```bash
python3 intranet/render_pages.py \
  --manifest /tmp/manifest.json \
  --renderer stdout \
  --limit 3 \
  --dwell-ms 100
```

### 3.7 内网一键执行

如果要一键生成 manifest 并开始展示：

```bash
bash intranet/run_sync.sh spc-cloud /data/repos/spc-cloud/spc-modules-system /tmp/out
```

参数说明：

- 第一个参数：项目名
- 第二个参数：本次扫描目录
- 第三个参数：输出目录

可选环境变量：

```bash
export CONFIG_PATH=config/projects.json
export BRANCH=feature/demo
export COMMIT=abc1234
export PAGE_LINES=40
export PAGE_COLS=110
export DWELL_MS=1800
```

然后执行：

```bash
bash intranet/run_sync.sh spc-cloud /data/repos/spc-cloud/spc-modules-system /tmp/out
```

### 3.8 内网当前支持的文件类型

文本型文件：

- `.java`
- `.xml`
- `.sql`
- `.properties`
- `.yml`
- `.yaml`
- `.sh`
- `.md`
- `.txt`

仅索引不做 OCR 正文恢复的二进制文件：

- `.class`
- `.jar`

### 3.9 内网使用建议

- `project_root` 固定为整个仓库根目录
- `scan_root` 根据这次要扫描的模块灵活切换
- 尽量保持页面分辨率、字体、窗口大小不变
- 长 SQL / 长 XML 必须保留横向切片，不要开启自动换行

## 4. 外网环境怎么用

### 4.1 外网环境要求

建议环境：

- Windows
- Python 3.11 或 3.14 虚拟环境
- 已安装 `paddleocr`
- 已安装 `opencv-python`
- 已安装 `requests`
- 已安装 `websockets`
- OBS
- 采集卡已能把内网画面输入到 OBS

当前你已经在 Windows 外网机上准备好了 Python 虚拟环境，并且基础 OCR 依赖已安装成功。

### 4.2 外网安装依赖

在已激活的虚拟环境中执行：

```bash
pip install -r requirements-ocr.txt
```

### 4.3 外网侧解决什么问题

外网侧负责：

- 从 OBS 抓取截图
- 对截图做 OCR
- 解析页面中的页头元数据
- 提取正文中的代码行
- 生成初步重建结果

当前版本还属于骨架阶段，已经打通了基础入口，但还没有把“复杂纠错 + 增量同步 + 最终文件覆盖”全部补完。

### 4.4 从 OBS 抓图

如果 OBS 已经打开，并且配置好了采集源，可以先单独抓图：

```bash
python external/obs_capture.py \
  --source "你的采集源名称" \
  --out-dir external_out/captures \
  --count 1
```

常用参数：

- `--host 127.0.0.1`
- `--port 4455`
- `--password ""`
- `--source "采集源名称"`
- `--image-format png`
- `--image-width 1920`
- `--count 5`
- `--interval-ms 1500`

示例：

```bash
python external/obs_capture.py \
  --host 127.0.0.1 \
  --port 4455 \
  --source "Video Capture Device" \
  --out-dir external_out/captures \
  --image-format png \
  --image-width 1920 \
  --count 3 \
  --interval-ms 1500
```

### 4.5 对单张截图运行 OCR

如果你已经有一张截图：

```bash
python external/ocr_runner.py \
  --image capture.png \
  --output external_out/ocr.json
```

输出内容包括：

- OCR 识别到的文本
- 每段文本的置信度
- 每段文本对应的区域坐标

### 4.6 解析代码页并生成初步重建结果

如果想直接走一条简单链路：

```bash
python external/sync_manager.py \
  --image capture.png \
  --workspace external_out
```

这一步会做：

1. OCR
2. 页头解析
3. 初步提取正文中的行号和代码文本
4. 生成报告文件

输出文件通常包括：

- `external_out/ocr.json`
- `external_out/rebuilt.json`
- `external_out/report.json`

### 4.7 单独校验重建结果

如果只想看重建结果里提取出了多少行：

```bash
python external/verifier.py \
  --rebuilt external_out/rebuilt.json
```

### 4.8 外网当前脚本作用说明

- `external/obs_capture.py`
  - 从 OBS 获取截图

- `external/page_detector.py`
  - 从 OCR 文本中解析：
    - `REPO`
    - `BRANCH`
    - `COMMIT`
    - `FILE`
    - `LANG`
    - `PAGE`
    - `LINES`
    - `SLICE`
    - `COLS`
    - `CHUNK`

- `external/ocr_runner.py`
  - 调用 PaddleOCR 扫描截图

- `external/code_rebuilder.py`
  - 从 OCR 文本中提取 `行号 + 代码内容`

- `external/sync_manager.py`
  - 串起 OCR、页头解析和重建

- `external/verifier.py`
  - 做最基础的结果检查

## 5. 当前推荐操作流程

### 5.1 第一步：内网扫描项目

```bash
python3 intranet/scan_project_tree.py \
  --project spc-cloud \
  --config config/projects.json \
  --scan-root /data/repos/spc-cloud/spc-modules-system \
  --output /tmp/project_scan.json
```

### 5.2 第二步：内网生成 manifest

```bash
python3 intranet/build_manifest.py \
  --project spc-cloud \
  --config config/projects.json \
  --scan-root /data/repos/spc-cloud/spc-modules-system \
  --output /tmp/manifest.json \
  --branch feature/demo \
  --commit abc1234
```

### 5.3 第三步：内网投屏

```bash
python3 intranet/render_pages.py \
  --manifest /tmp/manifest.json \
  --dwell-ms 1800
```

### 5.4 第四步：外网抓图或直接测试单张截图

```bash
python external/ocr_runner.py \
  --image capture.png \
  --output external_out/ocr.json
```

### 5.5 第五步：外网跑完整骨架流程

```bash
python external/sync_manager.py \
  --image capture.png \
  --workspace external_out
```

## 6. 当前状态与限制

当前已经完成：

- 多项目根目录配置
- 内网扫描目录树和文件类型
- 内网生成分页与横向切片 manifest
- 内网投屏显示
- 外网 OCR 骨架
- 外网页头解析骨架
- 外网初步重建骨架

当前还没完成：

- 基于 `git diff` 的增量页选择
- 从 OBS 连续自动抓页并去重
- XML / SQL / Java 的专项纠错器
- 完整的文件回写
- 最终镜像仓库同步

## 7. 下一步建议

当前最建议优先做的两件事：

1. 给外网补“截图裁切 + 页头区 / 正文区分开识别”
2. 给内网补“基于 `git diff` 只生成变更页”

这样一来，这套工具就能从“能跑骨架”进一步进入“可实际反复使用”的状态。
