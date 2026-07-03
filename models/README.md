# 模型目录说明

当前目录已经内含一份可直接使用的深度权重：

```text
models/
└─ depth_pro.pt
```

说明：

- `scripts/run_depthpro_demo.py` 直接读取这个 `depth_pro.pt`
- 这是当前交付包内默认附带的深度模型权重
- 如果后续要切换到 Hugging Face 风格的 metric depth 模型，可以再额外放一个模型目录，例如：

```text
models/
├─ depth_pro.pt
└─ depth-anything-metric-small/
   ├─ config.json
   ├─ preprocessor_config.json
   ├─ model.safetensors / pytorch_model.bin
   └─ ...
```

补充：

- `run_metric_depth_demo.py` 需要的是**目录**
- `run_depthpro_demo.py` 需要的是**单个 checkpoint 文件**
