# 后端接口

给小程序后端准备的 FastAPI 包装层

## 启动方式（我这里硬编码了，之后会改）

```powershell
pip install -r requirements.txt
$env:FOOD_VOLUME_DEPTHPRO_CHECKPOINT="D:\.PROJECTS\CV-FOOD-DETECT\2026-7-03\miniapp_food_volume_bundle\models\depth_pro.pt"
$env:FOOD_VOLUME_TIMEOUT_SECONDS="10"
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

## 图接口

接口：

```text
POST /api/v1/food-volume
Content-Type: multipart/form-data
```

字段：

```text
image        必填，用户拍摄的 RGB 图片
food         可选，食物类别，例如 strawberry / banana
phone_model  可选，手机型号，用于匹配相机参数
prompt       可选，自动分割文本提示
mask         可选，调试阶段可上传二值 mask
intrinsics   可选，调试阶段可上传相机内参 JSON
```

正式小程序端不需要上传 `mask`。如果没有传入 `mask`，后端会自动生成：

```text
优先：GroundingDINO 文本定位 -> SAM/MobileSAM 分割
替代：HuggingFace GroundingDINO 文本定位 -> GrabCut 前景分割
兜底：OpenCV 颜色阈值 -> GrabCut 前景分割
```

`mask` 字段只作为调试入口，用于和自动分割结果做对照。

## 后端返回

主要字段：

```text
request_id        本次请求 ID
elapsed_seconds   本次推理耗时
output_dir        后端保存中间结果的位置
pipeline_summary  主链路摘要
volume_summary    体积、质量、热量估计结果
```

`pipeline_summary.auto_mask` 会记录实际使用的 mask 生成方式。`pipeline_summary.timings_seconds` 会记录自动分割、DepthPro、体积计算和总耗时，后续可直接用于评估 10 秒目标。

## 饭前饭后摄入量估计接口

接口：

```text
POST /api/v1/food-consumption
Content-Type: multipart/form-data
```

字段：

```text
before_image       必填，饭前 RGB 图片
after_image        必填，饭后 RGB 图片
food               可选，食物类别，例如 strawberry / banana
phone_model        可选，手机型号，用于匹配相机参数
prompt             可选，自动分割文本提示
before_mask        可选，饭前二值 mask，仅调试阶段使用
after_mask         可选，饭后二值 mask，仅调试阶段使用
before_intrinsics  可选，饭前相机内参 JSON，仅调试阶段使用
after_intrinsics   可选，饭后相机内参 JSON，仅调试阶段使用
```

后端会分别调用两次单图估计链路，然后计算：

```text
consumed_volume_ml = before.estimated_volume_ml - after.estimated_volume_ml
consumed_mass_g    = before.estimated_mass_g - after.estimated_mass_g
consumed_kcal      = before.estimated_kcal - after.estimated_kcal
```

如果饭后估计值大于饭前估计值，接口会将摄入量截断为 0，并在 `consumed.warnings` 中记录原因，避免小程序展示负摄入量。

主要返回字段：

```text
request_id       本次请求 ID
elapsed_seconds  两张图片总耗时
before           饭前单图估计结果
after            饭后单图估计结果
consumed         摄入体积、质量、热量和 intake_ratio
output_dir       后端保存中间结果的位置
```
