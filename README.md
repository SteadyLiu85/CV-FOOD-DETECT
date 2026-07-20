# 小程序食物体积估计

## 目的

- 输入：`RGB 图像`
- 补充：`food class / 手机型号 / 调试用 mask / 调试用相机内参`
- 中间过程：`metric depth -> scaffold -> 支撑平面定尺 / 先验兜底`
- 输出：`体积 + 千卡 + 摘要 json + 可视化结果`

## 包含内容

### 脚本

- `scripts/run_metric_depth_demo.py`
  - 调用本地 metric depth 模型目录，对单张图做深度预测
  - 输出深度可视化、`depth_metric_mm.png` 和 scaffold 摘要

- `scripts/run_depthpro_demo.py`
  - Apple DepthPro 分支
  - 支持直接读取已有 intrinsics，或通过手机型号 / EXIF / DepthPro 焦距预测自动回填 intrinsics

- `scripts/infer_intrinsics_from_image.py`
  - 从手机型号配置、图片 EXIF 或焦距兜底值推断 intrinsics

- `scripts/run_grounded_sam_to_voleta.py`
  - 文本提示词驱动的自动食物 mask 生成入口
  - 依赖 GroundingDINO + MobileSAM 对应 vendor 与权重

- `scripts/auto_food_mask.py`
  - 小程序链路使用的自动 mask 入口
  - 优先尝试 GroundingDINO + MobileSAM；也支持 HuggingFace GroundingDINO 找框 + GrabCut
  - 若模型依赖或权重缺失，自动降级为 OpenCV 颜色阈值 + GrabCut 兜底
  - 目标是保证用户端不需要手动上传 mask

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

当前脚本会优先尝试从以下位置自动加载：

- 同级的 `ml-depth-pro/src`
- 内部的 `vendor/ml-depth-pro/src`

### `run_metric_depth_demo.py` 所需

需要一个本地深度模型目录，能够兼容：

```python
AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
AutoModelForDepthEstimation.from_pretrained(model_dir, local_files_only=True)
```

### `run_depthpro_demo.py` 所需

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
- `torchvision`
- `timm`
- `pillow_heif`
- `matplotlib`
- `transformers`

## 输入格式要求

### 1. 图像

- 单张 RGB 图像

### 2. Mask

- 用户端不需要提供 mask。
- 后端会通过 `run_auto_food_volume_demo.py` 自动生成 mask。
- 调试阶段仍保留手动传入二值 mask 的能力，用于和自动分割结果做对照。
- 自动 mask 默认：

```text
优先：GroundingDINO 文本定位 -> SAM/MobileSAM 分割
替代：HuggingFace GroundingDINO 文本定位 -> GrabCut 前景分割
else：OpenCV 颜色阈值 -> GrabCut 前景分割
```

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
- 最后使用 DepthPro 预测焦距

## 用法

### 自动输入

只给图片时，直接：

```powershell
python scripts/run_auto_food_volume_demo.py `
  --checkpoint models/depth_pro.pt `
  --image examples/input/example_image.jpg `
  --food strawberry `
  --output-dir runs/example_auto_demo
```

知道手机型号，可以补充：

```powershell
python scripts/run_auto_food_volume_demo.py `
  --checkpoint models/depth_pro.pt `
  --image examples/input/example_image.jpg `
  --food strawberry `
  --phone-model "EXAMPLE PHONE MODEL" `
  --phone-profiles assets/phone_camera_profiles.example.json `
  --output-dir runs/example_auto_demo
```

如果只想单独测试自动 mask：

```powershell
python scripts/auto_food_mask.py `
  --image examples/input/example_image.jpg `
  --food strawberry `
  --output-mask runs/example_auto_mask/mask.png `
  --metadata runs/example_auto_mask/mask.metadata.json
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

### 单图体积 / 热量估计

建议使用如下接口：

### 请求

- `image`
- 可选 `food class`
- 可选 `phone model`
- 可选 `mask`，仅调试使用，正式小程序端不需要传
- 可选 `intrinsics`

### 返回

- `estimated_volume_ml`
- `estimated_mass_g`
- `estimated_kcal`
- 可选调试图像 / json 

### 云端 FastAPI 入口

```powershell
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

接口：

```text
POST /api/v1/food-volume
```

该接口内部调用 `scripts/run_auto_food_volume_demo.py`，输出与命令行保持一致。

`pipeline_summary.json` 会记录分段耗时：

```text
auto_mask_seconds
depthpro_seconds
volume_seconds
total_seconds
```

### 饭前饭后摄入量估计

小程序可以上传饭前和饭后两张图：

```text
POST /api/v1/food-consumption
```

请求字段：

```text
before_image  饭前图片，必填
after_image   饭后图片，必填
food          食物类别，可选
phone_model   手机型号，可选
prompt        自动分割提示词，可选
```

调试阶段也可以分别传入：

```text
before_mask / after_mask
before_intrinsics / after_intrinsics
```

上述 `before_mask / after_mask` 只用于调试对照，正式用户流程仍然只上传饭前和饭后图片。

后端会对饭前、饭后图片各跑一次单图链路，然后输出差值：

```text
consumed_volume_ml = before_volume_ml - after_volume_ml
consumed_mass_g    = before_mass_g - after_mass_g
consumed_kcal      = before_kcal - after_kcal
```

返回结果中 `before` 和 `after` 保留两次完整估计结果，`consumed` 给出实际摄入体积、质量、热量和进食比例。若饭后估计值异常大于饭前估计值，接口会把摄入量截断为 0，并在 `consumed.warnings` 中保留提示。

## 目录结构

```text
miniapp_food_volume_bundle/
├─ assets/
│  └─ noref_food_priors.json
│  └─ phone_camera_profiles.example.json
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
│  ├─ auto_food_mask.py
│  ├─ infer_intrinsics_from_image.py
│  ├─ run_auto_food_volume_demo.py
│  ├─ run_depthpro_demo.py
│  ├─ run_grounded_sam_to_voleta.py
│  └─ run_metric_depth_demo.py
├─ .gitignore
├─ README.md
└─ requirements.txt
```
