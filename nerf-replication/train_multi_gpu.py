#!/usr/bin/env python3
"""
多GPU NeRF训练脚本
使用方法: python train_multi_gpu.py --cfg_file configs/nerf/lego_multi_gpu.yaml
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
from src.config import cfg, args
from src.utils.data_utils import to_cuda
from src.datasets import make_data_loader
from src.models import make_network
from src.train import make_trainer, make_optimizer, make_scheduler, make_recorder
from src.evaluators import make_evaluator


def setup_distributed():
    """设置分布式训练环境"""
    if cfg.distributed:
        # 分布式训练设置
        torch.cuda.set_device(cfg.local_rank)
        dist.init_process_group(backend='nccl')
        print(f"分布式训练初始化完成 - 本地排名: {cfg.local_rank}")
    else:
        # 单机多卡设置
        cfg.local_rank = 0
        print("单机多卡模式")


def main():
    # 设置分布式环境
    setup_distributed()
    
    # 显示GPU信息
    if cfg.local_rank == 0 or not cfg.distributed:
        print("=== 系统信息 ===")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"配置GPU: {cfg.gpus}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU #{i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # 创建数据集
    print("创建训练数据集...")
    train_loader = make_data_loader(cfg, is_train=True)
    
    print("创建测试数据集...")
    test_loader = make_data_loader(cfg, is_train=False)
    
    # 创建网络
    print("创建网络...")
    network = make_network(cfg)
    
    # 创建训练器
    print("创建训练器...")
    trainer = make_trainer(network)
    
    # 创建优化器
    print("创建优化器...")
    optimizer = make_optimizer(network, cfg.train)
    
    # 创建调度器
    print("创建学习率调度器...")
    scheduler = make_scheduler(optimizer, cfg.train.scheduler)
    
    # 创建记录器
    print("创建记录器...")
    recorder = make_recorder(cfg)
    
    # 创建评估器
    print("创建评估器...")
    evaluator = make_evaluator(cfg)
    
    # 开始训练
    print("开始训练...")
    for epoch in range(cfg.train.epoch):
        # 训练一个epoch
        trainer.train(epoch, train_loader, optimizer, recorder)
        
        # 更新学习率
        scheduler.step()
        
        # 验证
        if (epoch + 1) % cfg.eval_ep == 0:
            print(f"开始验证 epoch {epoch}...")
            trainer.val(epoch, test_loader, evaluator, recorder)
        
        # 保存模型
        if (epoch + 1) % cfg.save_ep == 0:
            if cfg.local_rank == 0 or not cfg.distributed:
                print(f"保存模型 epoch {epoch}...")
                recorder.save_model(network, optimizer, epoch)
        
        # 保存最新模型
        if (epoch + 1) % cfg.save_latest_ep == 0:
            if cfg.local_rank == 0 or not cfg.distributed:
                print(f"保存最新模型 epoch {epoch}...")
                recorder.save_model(network, optimizer, epoch, is_latest=True)
    
    print("训练完成!")
    
    # 清理分布式环境
    if cfg.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main() 