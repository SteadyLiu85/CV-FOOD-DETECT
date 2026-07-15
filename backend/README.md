# 云端后端接口说明

本目录给小程序后端准备一个最小 FastAPI 包装层。它不替代现有算法脚本，只负责把 HTTP 上传请求转换为本地推理命令。

## 启动方式

```powershell
pip install -r requirements.txt
$env:FOOD_VOLUME_DEPTHPRO_CHECKPOINT="D:\.PROJECTS\CV-FOOD-DETECT\2026-7-03\miniapp_food_volume_bundle\models\depth_pro.pt"
$env:FOOD_VOLUME_TIMEOUT_SECONDS="10"
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

## 前端请求

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

## 后端返回

主要字段：

```text
request_id        本次请求 ID
elapsed_seconds   本次推理耗时
output_dir        后端保存中间结果的位置
pipeline_summary  主链路摘要
volume_summary    体积、质量、热量估计结果
```

`pipeline_summary.timings_seconds` 会记录自动分割、DepthPro、体积计算和总耗时，后续可直接用于评估 10 秒目标。

## 当前边界

- 小程序前端只负责上传图片、可选食物类别、可选手机型号。
- 云端后端负责 mask、相机内参、DepthPro、3D scaffold、体积和热量计算。
- 当前无 mask 输入时仍依赖自动分割模型及其 vendor 代码，部署云端时需要把 GroundingDINO / SAM 或替代模型一并补齐。
- `FOOD_VOLUME_TIMEOUT_SECONDS` 用于控制服务级超时，本周目标是热启动推理控制在 10 秒左右。
