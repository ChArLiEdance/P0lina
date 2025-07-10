#!/usr/bin/env python3
"""
运行脚本，支持选择使用NeRF或KiloNeRF网络
使用方法:
    python run_kilonerf.py --network_type nerf --cfg_file configs/nerf/lego.yaml
    python run_kilonerf.py --network_type kilonerf --cfg_file configs/nerf/lego_kilonerf.yaml
"""

import argparse
import os
import sys
from src.config import cfg, args
from src.models import make_network
from src.train import (
    make_trainer,
    make_optimizer,
    make_lr_scheduler,
    make_recorder,
    set_lr_scheduler,
)
from src.datasets import make_data_loader
from src.utils.net_utils import (
    load_model,
    save_model,
    load_network,
    save_trained_config,
    load_pretrain,
)
from src.evaluators import make_evaluator
import torch
import torch.distributed as dist

torch.autograd.set_detect_anomaly(True)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='NeRF/KiloNeRF Training')
    parser.add_argument('--network_type', type=str, default='nerf', 
                       choices=['nerf', 'kilonerf'],
                       help='选择网络类型: nerf 或 kilonerf')
    parser.add_argument('--cfg_file', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('--test', action='store_true',
                       help='运行测试模式')
    parser.add_argument('--opts', nargs='*', default=[],
                       help='其他配置选项')
    
    return parser.parse_args()

def train(cfg, network):
    """训练函数"""
    if cfg.local_rank == 0:
        print("=" * 80)
        print("开始训练")
        print(f"网络类型: {getattr(cfg.task_arg, 'network_type', 'nerf')}")
        print("=" * 80)
        print(f"任务: {cfg.task}")
        print(f"场景: {cfg.scene}")
        print(f"实验名称: {cfg.exp_name}")
        print(f"总训练轮数: {cfg.train.epoch}")
        print(f"每轮迭代数: {cfg.ep_iter}")
        print(f"学习率: {cfg.train.lr}")
        print(f"批次大小: {cfg.train.batch_size}")
        print(f"光线数量: {cfg.task_arg.N_rays}")
        print(f"采样点数: {cfg.task_arg.N_samples}")
        print(f"使用视角信息: {cfg.task_arg.use_viewdirs}")
        print("=" * 80)

    save_trained_config(cfg)
    train_loader = make_data_loader(
        cfg, is_train=True, is_distributed=cfg.distributed, max_iter=cfg.ep_iter
    )
    val_loader = make_data_loader(cfg, is_train=False)

    trainer = make_trainer(cfg, network, train_loader)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)
    evaluator = make_evaluator(cfg)

    begin_epoch = load_model(
        network,
        optimizer,
        scheduler,
        recorder,
        cfg.trained_model_dir,
        resume=cfg.resume,
    )
    if begin_epoch == 0 and cfg.pretrain != "":
        load_pretrain(network, cfg.pretrain)

    set_lr_scheduler(cfg, scheduler)

    for epoch in range(begin_epoch, cfg.train.epoch):
        recorder.epoch = epoch
        if cfg.distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)

        train_loader.dataset.epoch = epoch

        if cfg.local_rank == 0:
            print(f"\n开始第 {epoch + 1}/{cfg.train.epoch} 轮训练")
            print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
            
        trainer.train(epoch, train_loader, optimizer, recorder)
        scheduler.step()

        if (epoch + 1) % cfg.save_ep == 0 and cfg.local_rank == 0:
            save_model(
                network, optimizer, scheduler, recorder, cfg.trained_model_dir, epoch
            )

        if (epoch + 1) % cfg.save_latest_ep == 0 and cfg.local_rank == 0:
            save_model(
                network,
                optimizer,
                scheduler,
                recorder,
                cfg.trained_model_dir,
                epoch,
                last=True,
            )

        if (epoch + 1) % cfg.eval_ep == 0 and cfg.local_rank == 0:
            trainer.val(epoch, val_loader, evaluator, recorder)

    return network

def test(cfg, network):
    """测试函数"""
    trainer = make_trainer(cfg, network)
    val_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    epoch = load_network(
        network, cfg.trained_model_dir, resume=cfg.resume, epoch=cfg.test.epoch
    )
    trainer.val(epoch, val_loader, evaluator)

def synchronize():
    """分布式训练同步函数"""
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def main():
    """主函数"""
    # 解析命令行参数
    cmd_args = parse_args()
    
    # 设置配置文件
    args.cfg_file = cmd_args.cfg_file
    args.test = cmd_args.test
    args.opts = cmd_args.opts
    
    # 更新配置
    cfg_ = cfg.make_cfg(args)
    
    # 设置网络类型
    if hasattr(cfg_.task_arg, 'network_type'):
        cfg_.task_arg.network_type = cmd_args.network_type
        print(f"设置网络类型为: {cmd_args.network_type}")
    
    # 分布式训练设置
    if cfg_.distributed:
        cfg_.local_rank = int(os.environ["RANK"]) % torch.cuda.device_count()
        torch.cuda.set_device(cfg_.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # 创建网络
    network = make_network(cfg_)
    
    # 运行训练或测试
    if cmd_args.test:
        test(cfg_, network)
    else:
        train(cfg_, network)
    
    if cfg_.local_rank == 0:
        print("Success!")
        print("=" * 80)
    
    os.system("kill -9 {}".format(os.getpid()))

if __name__ == "__main__":
    main() 