# 小程序食物体积估计

## 目的

当前链路：

- 输入：`RGB 图像`
- 可选补充：`food class / 手机型号 / 已有 mask / 已有相机内参`
- 中间过程：`metric depth -> scaffold -> 支撑平面定尺 / 先验兜底`
- 输出：`体积 + 千卡 + 摘要 json + 可视化结果`

## 包含内容

### 脚本

- `scripts/run_metric_depth_demo.py`
  - 调用本地 Hugging Face 风格的 metric depth 模型目录，对单张图做深度预测
  - 输出深度可视化、`depth_metric_mm.png` 和 scaffold 摘要

- `scripts/run_depthpro_demo.py`
  - Apple DepthPro 分支
  - 支持直接读取已有 intrinsics，或通过手机型号 / EXIF / DepthPro 焦距预测自动回填 intrinsics

- `scripts/infer_intrinsics_from_image.py`
  - 从手机型号配置、图片 EXIF 或焦距兜底值推断 intrinsics

- `scripts/run_grounded_sam_to_voleta.py`
  - 文本提示词驱动的自动食物 mask 生成入口
  - 依赖 GroundingDINO + MobileSAM 对应 vendor 与权重

- `scripts/run_auto_food_volume_demo.py`
  - 一键入口：自动 mask -> 自动 intrinsics -> DepthPro -> 体积估计

- `scripts/build_noref_scaffold.py`
  - 从 metric depth 或已有场景点云构建 3D scaffold 点集

- `scripts/estimate_metric_volume_calorie.py`
  - 基于 metric depth scaffold 估计体积和千卡

- `scripts/estimate_noref_scale_calorie.py`
  - 当 metric depth 不可用或不稳定时，使用先验估计

### 配置 / 先验

- `assets/noref_food_priors.json`
  - 存放若干单体食物的尺寸、密度和热量先验

- `assets/phone_camera_profiles.example.json`
  - 手机型号到相机参数的示例映射表

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

- 可以直接提供单通道二值 mask
- 也可以不提供，改由 `run_auto_food_volume_demo.py` 自动生成

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

如果不直接提供 intrinsics，也可以：

- 传手机型号并匹配 `assets/phone_camera_profiles.example.json`
- 让脚本读取图片 EXIF
- 最后使用 DepthPro 预测焦距兜底

## 推荐用法

### 一键自动输入方案

用户只给图片时，推荐直接走这一条：

```powershell
python scripts/run_auto_food_volume_demo.py `
  --checkpoint models/depth_pro.pt `
  --image examples/input/example_image.jpg `
  --food strawberry `
  --output-dir runs/example_auto_demo
```

如果你知道手机型号，也可以补充：

```powershell
python scripts/run_auto_food_volume_demo.py `
  --checkpoint models/depth_pro.pt `
  --image examples/input/example_image.jpg `
  --food strawberry `
  --phone-model "EXAMPLE PHONE MODEL" `
  --phone-profiles assets/phone_camera_profiles.example.json `
  --output-dir runs/example_auto_demo
```

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
- 可选 `food class`
- 可选 `phone model`
- 可选 `mask`
- 可选 `intrinsics`

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
│  └─ phone_camera_profiles.example.json
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
│  ├─ infer_intrinsics_from_image.py
│  ├─ run_auto_food_volume_demo.py
│  ├─ run_depthpro_demo.py
│  ├─ run_grounded_sam_to_voleta.py
│  └─ run_metric_depth_demo.py
├─ .gitignore
├─ README.md
└─ requirements.txt
```
