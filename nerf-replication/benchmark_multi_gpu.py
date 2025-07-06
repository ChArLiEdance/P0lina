#!/usr/bin/env python3
"""
多GPU性能测试脚本
用于比较单GPU和多GPU的训练性能
使用方法: python benchmark_multi_gpu.py --cfg_file configs/nerf/lego.yaml --iterations 100 --benchmark_type both
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from src.config import cfg, args
from src.datasets import make_data_loader
from src.models import make_network
from src.train import make_trainer
from src.utils.data_utils import to_cuda


def parse_benchmark_args():
    """解析benchmark相关的参数"""
    iterations = 100
    benchmark_type = "both"
    
    # 从args.opts中解析参数
    if hasattr(args, 'opts') and args.opts:
        i = 0
        while i < len(args.opts):
            if args.opts[i] == "--iterations" and i + 1 < len(args.opts):
                iterations = int(args.opts[i + 1])
                i += 2
            elif args.opts[i] == "--benchmark_type" and i + 1 < len(args.opts):
                benchmark_type = args.opts[i + 1]
                i += 2
            elif args.opts[i] == "--gpus" and i + 1 < len(args.opts):
                gpu_list = [int(x.strip()) for x in args.opts[i + 1].split(',')]
                cfg.gpus = gpu_list
                print(f"使用指定GPU: {cfg.gpus}")
                i += 2
            else:
                i += 1
    
    return iterations, benchmark_type


def benchmark_single_gpu(cfg, num_iterations=100):
    """单GPU性能测试"""
    print("=== 单GPU性能测试 ===")
    
    # 临时修改配置为单GPU
    original_gpus = cfg.gpus.copy()
    cfg.gpus = [0]  # 只使用一个GPU
    
    # 创建数据集
    train_loader = make_data_loader(cfg, is_train=True)
    
    # 创建网络
    network = make_network(cfg)
    
    # 创建训练器
    trainer = make_trainer(network)
    
    # 预热
    print("预热中...")
    for i, batch in enumerate(train_loader):
        if i >= 5:  # 预热5个batch
            break
        batch = to_cuda(batch, trainer.device)
        with torch.no_grad():
            _ = trainer.network(batch)
    
    # 性能测试
    print(f"开始性能测试 ({num_iterations} iterations)...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    total_loss = 0
    for i, batch in enumerate(train_loader):
        if i >= num_iterations:
            break
            
        batch = to_cuda(batch, trainer.device)
        batch["step"] = i
        
        # 前向传播
        with torch.no_grad():
            output, loss, loss_stats, image_stats = trainer.network(batch)
            total_loss += loss.item()
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    fps = num_iterations / total_time
    
    print(f"单GPU测试结果:")
    print(f"  总时间: {total_time:.2f}s")
    print(f"  平均batch时间: {avg_time:.4f}s")
    print(f"  FPS: {fps:.2f}")
    print(f"  平均loss: {total_loss/num_iterations:.6f}")
    
    # 恢复原始GPU配置
    cfg.gpus = original_gpus
    
    return avg_time, fps


def benchmark_multi_gpu(cfg, num_iterations=100):
    """多GPU性能测试"""
    print("=== 多GPU性能测试 ===")
    
    # 创建数据集
    train_loader = make_data_loader(cfg, is_train=True)
    
    # 创建网络
    network = make_network(cfg)
    
    # 创建训练器
    trainer = make_trainer(network)
    
    # 预热
    print("预热中...")
    for i, batch in enumerate(train_loader):
        if i >= 5:  # 预热5个batch
            break
        batch = to_cuda(batch, trainer.device)
        with torch.no_grad():
            _ = trainer.network(batch)
    
    # 性能测试
    print(f"开始性能测试 ({num_iterations} iterations)...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    total_loss = 0
    for i, batch in enumerate(train_loader):
        if i >= num_iterations:
            break
            
        batch = to_cuda(batch, trainer.device)
        batch["step"] = i
        
        # 前向传播
        with torch.no_grad():
            output, loss, loss_stats, image_stats = trainer.network(batch)
            total_loss += loss.item()
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    fps = num_iterations / total_time
    
    print(f"多GPU测试结果:")
    print(f"  总时间: {total_time:.2f}s")
    print(f"  平均batch时间: {avg_time:.4f}s")
    print(f"  FPS: {fps:.2f}")
    print(f"  平均loss: {total_loss/num_iterations:.6f}")
    
    return avg_time, fps


def main():
    # 解析benchmark参数
    iterations, benchmark_type = parse_benchmark_args()
    
    # 显示系统信息
    if cfg.local_rank == 0 or not cfg.distributed:
        print("=== 系统信息 ===")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"配置GPU: {cfg.gpus}")
        print(f"测试类型: {benchmark_type}")
        print(f"迭代次数: {iterations}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU #{i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # 根据测试类型执行测试
    single_time = None
    single_fps = None
    multi_time = None
    multi_fps = None
    
    if benchmark_type in ["single", "both"]:
        single_time, single_fps = benchmark_single_gpu(cfg, iterations)
    
    if benchmark_type in ["multi", "both"]:
        multi_time, multi_fps = benchmark_multi_gpu(cfg, iterations)
    
    # 性能对比
    if benchmark_type == "both" and single_time is not None and multi_time is not None:
        print("\n=== 性能对比 ===")
        speedup = single_time / multi_time
        efficiency = speedup / len(cfg.gpus) * 100
        
        print(f"单GPU平均时间: {single_time:.4f}s")
        print(f"多GPU平均时间: {multi_time:.4f}s")
        print(f"加速比: {speedup:.2f}x")
        print(f"并行效率: {efficiency:.1f}%")
        print(f"单GPU FPS: {single_fps:.2f}")
        print(f"多GPU FPS: {multi_fps:.2f}")
        print(f"FPS提升: {multi_fps/single_fps:.2f}x")
        
        # 内存使用情况
        print("\n=== 内存使用情况 ===")
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU #{i}: 已分配 {memory_allocated:.2f}GB, 已保留 {memory_reserved:.2f}GB")
    
    # 清理
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main() 