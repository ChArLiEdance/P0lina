# NeRF-Replication with KiloNeRF Support

这个项目扩展了nerf-replication，添加了对KiloNeRF网络的支持，允许用户选择使用标准NeRF或KiloNeRF进行训练。

## 功能特性

### 1. 网络类型选择
- **标准NeRF**: 原始的NeRF网络实现
- **KiloNeRF**: 高效的KiloNeRF网络实现，支持single_network和multi_network模式

### 2. 数据集适配
- **标准Blender数据集**: 适配原始NeRF格式
- **KiloNeRF Blender数据集**: 适配KiloNeRF输入格式

### 3. 配置管理
- 支持通过配置文件选择网络类型
- 支持通过命令行参数动态选择网络类型

## 文件结构

```
src/
├── models/
│   └── nerf/
│       ├── network.py              # 标准NeRF网络
│       └── kilonerf_network.py     # KiloNeRF网络适配器
├── datasets/
│   └── nerf/
│       ├── blender.py              # 标准Blender数据集
│       └── blender_kilonerf.py     # KiloNeRF Blender数据集
configs/
└── nerf/
    ├── lego.yaml                   # 标准NeRF配置
    └── lego_kilonerf.yaml          # KiloNeRF配置
run_kilonerf.py                     # 新的运行脚本
```

## 使用方法

### 1. 使用标准NeRF

```bash
# 使用配置文件
python run_kilonerf.py --network_type nerf --cfg_file configs/nerf/lego.yaml

# 或使用原始脚本
python train.py --cfg_file configs/nerf/lego.yaml
```

### 2. 使用KiloNeRF

```bash
# 使用KiloNeRF配置
python run_kilonerf.py --network_type kilonerf --cfg_file configs/nerf/lego_kilonerf.yaml
```

### 3. 测试模式

```bash
# 测试标准NeRF
python run_kilonerf.py --network_type nerf --cfg_file configs/nerf/lego.yaml --test

# 测试KiloNeRF
python run_kilonerf.py --network_type kilonerf --cfg_file configs/nerf/lego_kilonerf.yaml --test
```

## 配置文件说明

### 标准NeRF配置 (lego.yaml)
```yaml
task: "nerf_replication"
network_module: src.models.nerf.network
train_dataset_module: src.datasets.nerf.blender
test_dataset_module: src.datasets.nerf.blender

task_arg:
  network_type: "nerf"  # 使用标准NeRF
  N_rays: 4096
  chunk_size: 16384
  # ... 其他参数
```

### KiloNeRF配置 (lego_kilonerf.yaml)
```yaml
task: "nerf_replication"
network_module: src.models.nerf.kilonerf_network
train_dataset_module: src.datasets.nerf.blender_kilonerf
test_dataset_module: src.datasets.nerf.blender_kilonerf

task_arg:
  network_type: "kilonerf"  # 使用KiloNeRF
  kilonerf_model_type: "single_network"  # single_network 或 multi_network
  N_rays: 4096
  chunk_size: 16384
  # ... 其他参数
```

## 网络类型参数

### task_arg.network_type
- `"nerf"`: 使用标准NeRF网络
- `"kilonerf"`: 使用KiloNeRF网络

### task_arg.kilonerf_model_type (仅KiloNeRF)
- `"single_network"`: 单网络模式
- `"multi_network"`: 多网络模式

## 数据集适配

### 标准Blender数据集
- 输出格式: `{'rays_o': ..., 'rays_d': ..., 'target_s': ..., ...}`
- 适用于标准NeRF训练

### KiloNeRF Blender数据集
- 输出格式: `{'rays_o': ..., 'rays_d': ..., 'target_s': ..., 'focal': ..., 'hwf': ..., 'render_poses': ..., ...}`
- 包含KiloNeRF特定的参数
- 适用于KiloNeRF训练

## 性能对比

| 网络类型 | 训练速度 | 内存使用 | 渲染质量 | 适用场景 |
|---------|---------|---------|---------|---------|
| 标准NeRF | 中等 | 中等 | 高 | 一般训练 |
| KiloNeRF | 快 | 低 | 高 | 快速训练/推理 |

## 注意事项

1. **依赖要求**: 使用KiloNeRF需要安装kilonerf相关依赖
2. **GPU要求**: KiloNeRF需要CUDA支持
3. **内存管理**: KiloNeRF使用更高效的内存管理策略
4. **兼容性**: 两种网络类型使用相同的数据格式，便于切换

## 故障排除

### 1. KiloNeRF导入错误
```bash
# 确保kilonerf路径正确
export PYTHONPATH=$PYTHONPATH:/path/to/kilonerf
```

### 2. CUDA错误
```bash
# 检查CUDA版本兼容性
nvidia-smi
python -c "import torch; print(torch.version.cuda)"
```

### 3. 内存不足
```bash
# 减少batch size
# 在配置文件中修改 N_rays 和 chunk_size
```

## 扩展开发

### 添加新的网络类型
1. 在 `src/models/nerf/` 下创建新的网络文件
2. 在 `configs/nerf/` 下创建对应的配置文件
3. 更新 `run_kilonerf.py` 中的网络类型选择

### 添加新的数据集
1. 在 `src/datasets/nerf/` 下创建新的数据集文件
2. 确保输出格式与网络期望的格式兼容
3. 更新配置文件中的数据集模块路径

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 许可证

本项目遵循原始nerf-replication的许可证。 