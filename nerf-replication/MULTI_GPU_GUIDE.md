# 多GPU NeRF训练指南

本指南介绍如何使用多GPU加速NeRF训练，确保并行速度大于单GPU速度。

## 功能特性

- **自动GPU检测**: 自动检测可用GPU数量
- **智能并行策略**: 根据GPU数量自动选择最优并行策略
- **性能优化**: 自动调整batch size、chunk size等参数
- **实时监控**: 显示训练速度、加速比等性能指标
- **内存管理**: 智能内存分配和清理

## 系统要求

- PyTorch >= 1.8.0
- CUDA >= 11.0
- 多GPU环境（推荐2-4张GPU）

## 快速开始

### 1. 性能测试

首先测试多GPU性能：

```bash
# 测试单GPU性能
python benchmark_multi_gpu.py --cfg_file configs/nerf/lego.yaml --gpus 0 --iterations 100

# 测试多GPU性能
python benchmark_multi_gpu.py --cfg_file configs/nerf/lego_multi_gpu.yaml --gpus 0,1,2,3 --iterations 100

# 只测试单GPU
python benchmark_multi_gpu.py --cfg_file configs/nerf/lego.yaml --benchmark_type single --iterations 100

# 只测试多GPU
python benchmark_multi_gpu.py --cfg_file configs/nerf/lego_multi_gpu.yaml --benchmark_type multi --iterations 100
```

### 2. 开始训练

使用多GPU配置文件进行训练：

```bash
# 使用多GPU配置文件
python train_multi_gpu.py --cfg_file configs/nerf/lego_multi_gpu.yaml

# 使用原始训练脚本（也会自动检测多GPU）
python train.py --cfg_file configs/nerf/lego_multi_gpu.yaml

# 指定特定GPU（通过修改配置文件中的gpus参数）
# 或者使用环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
python train_multi_gpu.py --cfg_file configs/nerf/lego_multi_gpu.yaml
```

## 配置说明

### 多GPU优化配置 (lego_multi_gpu.yaml)

```yaml
task_arg:
  N_rays: 4096        # 增加ray数量以充分利用多GPU
  chunk_size: 16384   # 增加chunk size以提高并行效率
  # ... 其他参数

train:
  num_workers: 8      # 增加worker数量以配合多GPU
  # ... 其他参数

log_interval: 20      # 减少log间隔以更好地监控性能
```

### GPU配置

```yaml
gpus: [0, 1, 2, 3]    # 指定使用的GPU设备
```

## 性能优化策略

### 1. 自动参数调整

系统会根据GPU数量自动调整以下参数：

- **N_rays**: 根据GPU数量线性增加
- **chunk_size**: 根据GPU数量调整以避免内存不足
- **num_workers**: 增加数据加载并行度

### 2. 并行策略选择

- **单GPU**: 标准训练模式
- **多GPU (DataParallel)**: 单机多卡，适合大多数场景
- **分布式 (DistributedDataParallel)**: 多机多卡，适合大规模训练

### 3. 内存优化

- 自动调整chunk size避免OOM
- 智能内存分配
- 定期内存清理

## 性能监控

训练过程中会显示以下性能指标：

```
eta: 1:23:45  epoch: 1  step: 100  loss: 0.1234  data: 0.0012  batch: 0.0456  speedup: 3.85x  lr: 0.000500  max_mem: 2048MB
```

- **speedup**: 相对于单GPU的加速比
- **batch**: 平均batch处理时间
- **max_mem**: 最大内存使用量

## 常见问题

### Q: 多GPU训练速度不如单GPU？

**A**: 可能的原因和解决方案：

1. **数据加载瓶颈**: 增加 `num_workers`
2. **GPU间通信开销**: 减少GPU数量或使用更快的连接
3. **内存不足**: 减少 `chunk_size` 或 `N_rays`
4. **模型太小**: 对于小模型，多GPU开销可能超过收益

### Q: 出现OOM错误？

**A**: 解决方案：

1. 减少 `chunk_size`
2. 减少 `N_rays`
3. 减少 `N_samples` 或 `N_importance`
4. 使用更少的GPU

### Q: 如何选择最优GPU数量？

**A**: 建议：

1. 先测试2GPU，观察加速比
2. 逐步增加GPU数量，找到最佳平衡点
3. 通常4GPU以内效果较好，超过4GPU收益递减

## 性能基准

在典型配置下的预期性能：

| GPU数量 | 加速比 | 并行效率 | 内存使用 |
|---------|--------|----------|----------|
| 1       | 1.0x   | 100%     | 2-4GB    |
| 2       | 1.8x   | 90%      | 4-8GB    |
| 4       | 3.2x   | 80%      | 8-16GB   |
| 8       | 5.5x   | 69%      | 16-32GB  |

*注：实际性能取决于硬件配置和模型大小*

## 高级配置

### 自定义GPU选择

```bash
# 修改配置文件中的gpus参数
# 或者使用环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
python train_multi_gpu.py --cfg_file configs/nerf/lego_multi_gpu.yaml
```

### 分布式训练

```bash
# 单机多卡分布式训练
python -m torch.distributed.launch --nproc_per_node=4 train_multi_gpu.py --cfg_file configs/nerf/lego_multi_gpu.yaml
```

## 故障排除

### 1. CUDA版本不匹配

```bash
# 检查CUDA版本
nvidia-smi
python -c "import torch; print(torch.version.cuda)"
```

### 2. GPU内存不足

```bash
# 监控GPU使用情况
watch -n 1 nvidia-smi
```

### 3. 性能分析

```bash
# 使用性能分析工具
python -m torch.utils.bottleneck train_multi_gpu.py --cfg_file configs/nerf/lego_multi_gpu.yaml
```

## 联系支持

如果遇到问题，请提供以下信息：

1. 系统配置（GPU型号、数量、内存）
2. PyTorch和CUDA版本
3. 错误日志
4. 性能测试结果

---

*最后更新: 2024年* 