# 小程序食物体积估计

## 目的

当前链路：

- 输入：`RGB 图像 + 食物 mask + 相机内参`
- 中间过程：`metric depth -> scaffold -> 支撑平面定尺 / 先验兜底`
- 输出：`体积 + 千卡 + 摘要 json + 可视化结果`

## 包含内容

### 脚本

- `scripts/run_metric_depth_demo.py`
  - 调用本地 Hugging Face 风格的 metric depth 模型目录，对单张图做深度预测
  - 输出深度可视化、`depth_metric_mm.png` 和 scaffold 摘要

- `scripts/run_depthpro_demo.py`
  - 备用分支，用于接 Apple DepthPro

- `scripts/build_noref_scaffold.py`
  - 从 metric depth 或已有场景点云构建 3D scaffold 点集

- `scripts/estimate_metric_volume_calorie.py`
  - 基于 metric depth scaffold 估计体积和千卡

- `scripts/estimate_noref_scale_calorie.py`
  - 当 metric depth 不可用或不稳定时，使用先验估计

### 配置 / 先验

- `assets/noref_food_priors.json`
  - 存放若干单体食物的尺寸、密度和热量先验

### 示例输入

- `examples/input/example_image.jpg`
- `examples/input/example_mask.jpg`
- `examples/input/intrinsics.json`

### 示例输出

- `examples/output_metric_demo/`
  - metric depth 分支的示例输出

- `examples/output_support_plane/`
  - 支撑平面体积估计的示例输出

## 模型文件说明

当前交付包已经内含一份可直接使用的 `DepthPro` 权重：

- `models/depth_pro.pt`

### `run_metric_depth_demo.py` 所需

需要一个本地 Hugging Face 风格的深度模型目录，能够兼容：

```python
AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
AutoModelForDepthEstimation.from_pretrained(model_dir, local_files_only=True)
```

原实验日志中使用的模型目录是：

```text
D:\models\depth-anything-metric-small
```

### `run_depthpro_demo.py` 所需

当前包内已经包含本地 checkpoint 文件：

```text
models/depth_pro.pt
```

见 `requirements.txt`。

核心依赖包括：

- `numpy`
- `Pillow`
- `scipy`
- `opencv-python`
- `torch`
- `transformers`

## 输入格式要求

### 1. 图像

- 单张 RGB 图像

### 2. Mask

- 单通道二值 mask
- 白色或非零区域表示食物区域

### 3. 相机内参

JSON 至少应包含：

```json
{
  "fl_x": 1501.7140890309136,
  "fl_y": 1501.7140890309136,
  "cx": 719.5,
  "cy": 960.5,
  "w": 1439.0,
  "h": 1921.0
}
```

## 推荐用法

### metric-depth 部分

第 1 步：运行 metric depth

```powershell
python scripts/run_metric_depth_demo.py `
  --model-dir <LOCAL_DEPTH_MODEL_DIR> `
  --image examples/input/example_image.jpg `
  --mask examples/input/example_mask.jpg `
  --intrinsics examples/input/intrinsics.json `
  --output-dir runs/example_metric_demo
```

第 2 步：估计支撑平面体积

```powershell
python scripts/estimate_metric_volume_calorie.py `
  --mode support_plane `
  --food strawberry `
  --metric-summary runs/example_metric_demo/02_results/metric_scaffold/summary.json `
  --output-dir runs/example_support_plane
```

### 先验部分

```powershell
python scripts/estimate_noref_scale_calorie.py `
  --summary examples/output_metric_demo/metric_scaffold_summary.json `
  --food strawberry `
  --output-dir runs/example_prior_fallback
```

## 输出文件

### Metric demo 

- `demo_summary.json`
- `02_results/depth_metric_mm.png`
- `02_results/metric_scaffold/summary.json`
- `02_results/metric_scaffold/overlay_projection.png`

### 体积估计

- `metric_volume_calorie_summary.json`
- `support_plane_overlay.png`
- `height_map.png`

## 小程序

建议使用如下接口：

### 请求

- `image`
- `mask`
- `food class`
- `intrinsics`

### 返回

- `estimated_volume_ml`
- `estimated_mass_g`
- `estimated_kcal`
- 可选调试图像 / json 

## 目录结构

```text
miniapp_food_volume_bundle/
├─ assets/
│  └─ noref_food_priors.json
├─ docs/
│  └─ 交付说明.md
├─ examples/
│  ├─ input/
│  │  ├─ example_image.jpg
│  │  ├─ example_mask.jpg
│  │  └─ intrinsics.json
│  ├─ output_metric_demo/
│  └─ output_support_plane/
├─ models/
│  └─ README.md
├─ scripts/
│  ├─ build_noref_scaffold.py
│  ├─ estimate_metric_volume_calorie.py
│  ├─ estimate_noref_scale_calorie.py
│  ├─ run_depthpro_demo.py
│  └─ run_metric_depth_demo.py
├─ .gitignore
├─ README.md
└─ requirements.txt
```
