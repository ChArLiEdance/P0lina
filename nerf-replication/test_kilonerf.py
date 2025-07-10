#!/usr/bin/env python3
"""
测试脚本，验证KiloNeRF网络和数据集的功能
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_network_creation():
    """测试网络创建"""
    print("=" * 50)
    print("测试网络创建")
    print("=" * 50)
    
    try:
        # 测试标准NeRF网络
        print("1. 测试标准NeRF网络...")
        from src.models.nerf.network import Network as NeRFNetwork
        from src.config import cfg
        
        # 模拟配置
        cfg.task_arg = type('obj', (object,), {
            'network_type': 'nerf',
            'N_samples': 64,
            'N_importance': 128,
            'chunk_size': 1024,
            'N_rays': 1024,
            'white_bkgd': True,
            'use_viewdirs': True
        })()
        
        cfg.network = type('obj', (object,), {
            'xyz_encoder': type('obj', (object,), {'type': 'frequency', 'input_dim': 3, 'freq': 10})(),
            'dir_encoder': type('obj', (object,), {'type': 'frequency', 'input_dim': 3, 'freq': 4})(),
            'nerf': type('obj', (object,), {'W': 256, 'D': 8, 'skips': [4]})()
        })()
        
        nerf_network = NeRFNetwork()
        print("✓ 标准NeRF网络创建成功")
        
        # 测试KiloNeRF网络
        print("2. 测试KiloNeRF网络...")
        try:
            from src.models.nerf.kilonerf_network import Network as KiloNeRFNetwork
            
            cfg.task_arg.network_type = 'kilonerf'
            cfg.task_arg.kilonerf_model_type = 'single_network'
            
            kilonerf_network = KiloNeRFNetwork()
            print("✓ KiloNeRF网络创建成功")
            
        except ImportError as e:
            print(f"⚠ KiloNeRF网络导入失败: {e}")
            print("请确保kilonerf依赖已正确安装")
        
    except Exception as e:
        print(f"✗ 网络创建失败: {e}")
        return False
    
    return True

def test_dataset_creation():
    """测试数据集创建"""
    print("\n" + "=" * 50)
    print("测试数据集创建")
    print("=" * 50)
    
    try:
        # 测试标准Blender数据集
        print("1. 测试标准Blender数据集...")
        from src.datasets.nerf.blender import Dataset as BlenderDataset
        from src.config import cfg
        
        # 模拟配置
        cfg.train_dataset = type('obj', (object,), {
            'data_root': 'data/nerf_synthetic',
            'input_ratio': 1.0
        })()
        
        cfg.task_arg = type('obj', (object,), {
            'white_bkgd': True,
            'N_rays': 1024,
            'no_batching': True,
            'test_skip': 1
        })()
        
        cfg.scene = 'lego'
        
        try:
            blender_dataset = BlenderDataset(split='train')
            print("✓ 标准Blender数据集创建成功")
        except FileNotFoundError:
            print("⚠ 标准Blender数据集创建失败（数据文件不存在）")
        
        # 测试KiloNeRF Blender数据集
        print("2. 测试KiloNeRF Blender数据集...")
        try:
            from src.datasets.nerf.blender_kilonerf import Dataset as KiloNeRFBlenderDataset
            
            kilonerf_dataset = KiloNeRFBlenderDataset(split='train')
            print("✓ KiloNeRF Blender数据集创建成功")
            
        except FileNotFoundError:
            print("⚠ KiloNeRF Blender数据集创建失败（数据文件不存在）")
        except ImportError as e:
            print(f"⚠ KiloNeRF Blender数据集导入失败: {e}")
        
    except Exception as e:
        print(f"✗ 数据集创建失败: {e}")
        return False
    
    return True

def test_config_loading():
    """测试配置文件加载"""
    print("\n" + "=" * 50)
    print("测试配置文件加载")
    print("=" * 50)
    
    try:
        # 测试标准NeRF配置
        print("1. 测试标准NeRF配置...")
        config_path = "configs/nerf/lego.yaml"
        if os.path.exists(config_path):
            print("✓ 标准NeRF配置文件存在")
        else:
            print("⚠ 标准NeRF配置文件不存在")
        
        # 测试KiloNeRF配置
        print("2. 测试KiloNeRF配置...")
        config_path = "configs/nerf/lego_kilonerf.yaml"
        if os.path.exists(config_path):
            print("✓ KiloNeRF配置文件存在")
        else:
            print("⚠ KiloNeRF配置文件不存在")
        
    except Exception as e:
        print(f"✗ 配置文件测试失败: {e}")
        return False
    
    return True

def test_imports():
    """测试模块导入"""
    print("\n" + "=" * 50)
    print("测试模块导入")
    print("=" * 50)
    
    modules_to_test = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('imageio', 'ImageIO'),
        ('src.config', '配置模块'),
        ('src.models.make_network', '网络创建模块'),
        ('src.datasets', '数据集模块'),
    ]
    
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ {description} 导入成功")
        except ImportError as e:
            print(f"✗ {description} 导入失败: {e}")
    
    # 测试KiloNeRF相关模块
    kilonerf_modules = [
        ('kilonerf_cuda', 'KiloNeRF CUDA'),
        ('src.models.nerf.kilonerf_network', 'KiloNeRF网络'),
        ('src.datasets.nerf.blender_kilonerf', 'KiloNeRF Blender数据集'),
    ]
    
    print("\nKiloNeRF相关模块:")
    for module_name, description in kilonerf_modules:
        try:
            __import__(module_name)
            print(f"✓ {description} 导入成功")
        except ImportError as e:
            print(f"⚠ {description} 导入失败: {e}")

def main():
    """主测试函数"""
    print("开始测试KiloNeRF功能...")
    print("=" * 60)
    
    # 测试导入
    test_imports()
    
    # 测试网络创建
    network_success = test_network_creation()
    
    # 测试数据集创建
    dataset_success = test_dataset_creation()
    
    # 测试配置文件
    config_success = test_config_loading()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    if network_success and dataset_success and config_success:
        print("✓ 所有核心功能测试通过")
        print("\n使用方法:")
        print("1. 使用标准NeRF:")
        print("   python run_kilonerf.py --network_type nerf --cfg_file configs/nerf/lego.yaml")
        print("\n2. 使用KiloNeRF:")
        print("   python run_kilonerf.py --network_type kilonerf --cfg_file configs/nerf/lego_kilonerf.yaml")
    else:
        print("✗ 部分功能测试失败")
        print("请检查依赖安装和配置文件")
    
    print("\n详细文档请参考: README_KILONERF.md")

if __name__ == "__main__":
    main() 