# 模型说明

```text
models/
└─ depth_pro.pt
```

说明：

- `scripts/run_depthpro_demo.py` 直接读取这个 `depth_pro.pt`
- 如有后续切换 metric depth 模型，可以再额外放一个模型目录，例如：

```text
models/
├─ depth_pro.pt
└─ depth-anything-metric-small/
   ├─ config.json
   ├─ preprocessor_config.json
   ├─ model.safetensors / pytorch_model.bin
   └─ ...
```
