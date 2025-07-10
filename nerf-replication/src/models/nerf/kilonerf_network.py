import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from src.models.encoding import get_encoder
from src.config import cfg
import sys
import os

# 添加kilonerf路径到sys.path
kilonerf_path = os.path.join(os.path.dirname(__file__), '../../../../kilonerf')
if kilonerf_path not in sys.path:
    sys.path.append(kilonerf_path)

# 导入kilonerf相关模块
try:
    from run_nerf_helpers import NeRF, get_embedder
    from utils import create_nerf
    from multi_modules import MultiNetwork
    from local_distill import create_multi_network_fourier_embedding, create_multi_network
    import kilonerf_cuda
except ImportError as e:
    print(f"Warning: Could not import kilonerf modules: {e}")
    print("Please ensure kilonerf is properly installed and accessible")


class KiloNeRF(nn.Module):
    """
    KiloNeRF网络适配器，仿照nerf-replication的格式
    支持single_network和multi_network两种模式
    """
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, skips=[4], use_viewdirs=False):
        super(KiloNeRF, self).__init__()
        """
        D：网络的深度，表示网络的层数
        W：网络每层的宽度，表示每层的神经元数目
        input_ch 和 input_ch_views：分别表示输入的点坐标和视角信息的维度
        skips：在这些层之间跳跃连接
        use_viewdirs：是否使用视角信息来影响输出
        """
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.output_ch = 5 if self.use_viewdirs else 4

        # 创建KiloNeRF网络
        self.model_type = getattr(cfg.task_arg, 'kilonerf_model_type', 'single_network')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.model_type == 'single_network':
            self._create_single_network()
        else:
            self._create_multi_network()
            
        print(f"KiloNeRF网络初始化完成:")
        print(f"  - 模型类型: {self.model_type}")
        print(f"  - 网络深度: {self.D}")
        print(f"  - 网络宽度: {self.W}")
        print(f"  - 使用视角信息: {self.use_viewdirs}")

    def _create_single_network(self):
        """创建单网络模式"""
        # 创建NeRF网络
        self.model = NeRF(
            D=self.D,
            W=self.W,
            input_ch=self.input_ch,
            input_ch_views=self.input_ch_views,
            skips=self.skips,
            use_viewdirs=self.use_viewdirs,
        )
        
        # 创建精细网络
        self.model_fine = NeRF(
            D=self.D,
            W=self.W,
            input_ch=self.input_ch,
            input_ch_views=self.input_ch_views,
            skips=self.skips,
            use_viewdirs=self.use_viewdirs,
        )
        
        self.model = self.model.to(self.device)
        self.model_fine = self.model_fine.to(self.device)

    def _create_multi_network(self):
        """创建多网络模式"""
        # 初始化kilonerf_cuda
        try:
            kilonerf_cuda.init_stream_pool(16)
            kilonerf_cuda.init_magma()
        except:
            print("Warning: kilonerf_cuda initialization failed")
        
        # 创建傅里叶嵌入
        position_num_input_channels, position_fourier_embedding = create_multi_network_fourier_embedding(1, 10)
        direction_num_input_channels, direction_fourier_embedding = create_multi_network_fourier_embedding(1, 4)
        
        # 创建多网络
        num_networks = 1  # 简化版本，使用单个网络
        self.multi_network = create_multi_network(
            num_networks, 
            position_num_input_channels, 
            direction_num_input_channels, 
            4,  # 输出通道数
            'multimatmul_differentiable', 
            cfg
        ).to(self.device)
        
        self.position_fourier_embedding = position_fourier_embedding
        self.direction_fourier_embedding = direction_fourier_embedding

    def batchify(self, fn, chunk):
        """Constructs a version of 'fn' that applies to smaller batches."""
        def ret(inputs):
            return torch.cat(
                [fn(inputs[i : i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0
            )
        return ret

    def forward(self, inputs, viewdirs=None, model=""):
        """前向传播"""
        if self.model_type == 'single_network':
            return self._forward_single_network(inputs, viewdirs, model)
        else:
            return self._forward_multi_network(inputs, viewdirs)

    def _forward_single_network(self, inputs, viewdirs=None, model=""):
        """单网络前向传播"""
        if model == "fine":
            fn = self.model_fine
        else:
            fn = self.model

        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        
        if self.use_viewdirs and viewdirs is not None:
            input_dirs = viewdirs[:, None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embedded = torch.cat([inputs_flat, input_dirs_flat], -1)
        else:
            embedded = inputs_flat

        embedded = embedded.to(torch.float32)
        chunk_size = getattr(cfg.task_arg, 'chunk_size', 1024)
        outputs_flat = self.batchify(fn, chunk_size)(embedded)
        outputs = torch.reshape(
            outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]
        )
        return outputs

    def _forward_multi_network(self, inputs, viewdirs=None):
        """多网络前向传播"""
        # 分离位置和方向信息
        if viewdirs is not None:
            positions, directions = torch.split(inputs, [self.input_ch, self.input_ch_views], dim=-1)
        else:
            positions = inputs
            directions = torch.zeros_like(positions)
        
        # 使用傅里叶嵌入
        if self.position_fourier_embedding is not None:
            embedded_positions = self.position_fourier_embedding(positions.unsqueeze(0)).squeeze(0)
        else:
            embedded_positions = positions
            
        if self.direction_fourier_embedding is not None:
            embedded_directions = self.direction_fourier_embedding(directions.unsqueeze(0)).squeeze(0)
        else:
            embedded_directions = directions
        
        # 合并嵌入
        embedded = torch.cat([embedded_positions, embedded_directions], -1)
        
        # 前向传播
        batch_size_per_network = torch.tensor([embedded.shape[0]], dtype=torch.long, device=self.device)
        outputs = self.multi_network(embedded, batch_size_per_network)
        
        return outputs


class Network(nn.Module):
    """
    主网络类，仿照nerf-replication的格式
    支持选择使用NeRF或KiloNeRF
    """
    def __init__(self):
        super(Network, self).__init__()
        
        # 获取网络类型参数
        self.network_type = getattr(cfg.task_arg, 'network_type', 'nerf')  # 'nerf' 或 'kilonerf'
        
        # 基本参数
        self.N_samples = cfg.task_arg.N_samples
        self.N_importance = cfg.task_arg.N_importance
        self.chunk = cfg.task_arg.chunk_size
        self.batch_size = cfg.task_arg.N_rays
        self.white_bkgd = cfg.task_arg.white_bkgd
        self.use_viewdirs = cfg.task_arg.use_viewdirs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 编码器
        self.embed_fn, self.input_ch = get_encoder(cfg.network.xyz_encoder)
        self.embeddirs_fn, self.input_ch_views = get_encoder(cfg.network.dir_encoder)

        # 根据网络类型创建模型
        if self.network_type == 'kilonerf':
            print("使用KiloNeRF网络")
            self.model = KiloNeRF(
                D=cfg.network.nerf.D,
                W=cfg.network.nerf.W,
                input_ch=self.input_ch,
                input_ch_views=self.input_ch_views,
                skips=cfg.network.nerf.skips,
                use_viewdirs=self.use_viewdirs,
            )
        else:
            print("使用标准NeRF网络")
            # 创建标准NeRF网络
            from .network import NeRF
            self.model = NeRF(
                D=cfg.network.nerf.D,
                W=cfg.network.nerf.W,
                input_ch=self.input_ch,
                input_ch_views=self.input_ch_views,
                skips=cfg.network.nerf.skips,
                use_viewdirs=self.use_viewdirs,
            )

            # 精细模型
            self.model_fine = NeRF(
                D=cfg.network.nerf.D,
                W=cfg.network.nerf.W,
                input_ch=self.input_ch,
                input_ch_views=self.input_ch_views,
                skips=cfg.network.nerf.skips,
                use_viewdirs=self.use_viewdirs,
            )

        print(f"Network初始化完成:")
        print(f"  - 网络类型: {self.network_type}")
        print(f"  - N_samples: {self.N_samples}")
        print(f"  - N_importance: {self.N_importance}")
        print(f"  - chunk_size: {self.chunk}")
        print(f"  - N_rays: {self.batch_size}")
        print(f"  - use_viewdirs: {self.use_viewdirs}")

        # 移动到设备
        if self.network_type == 'kilonerf':
            self.model = self.model.to(self.device)
        else:
            self.model = self.model.to(self.device)
            self.model_fine = self.model_fine.to(self.device)

    def batchify(self, fn, chunk):
        """Constructs a version of 'fn' that applies to smaller batches."""
        def ret(inputs):
            return torch.cat(
                [fn(inputs[i : i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0
            )
        return ret

    def forward(self, inputs, viewdirs, model=""):
        """Prepares inputs and applies network 'fn'."""
        if self.network_type == 'kilonerf':
            return self.model(inputs, viewdirs, model)
        else:
            # 标准NeRF前向传播
            if model == "fine":
                fn = self.model_fine
            else:
                fn = self.model

            inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
            embedded = self.embed_fn(inputs_flat)

            if self.use_viewdirs:
                input_dirs = viewdirs[:, None].expand(inputs.shape)
                input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
                embedded_dirs = self.embeddirs_fn(input_dirs_flat)
                embedded = torch.cat([embedded, embedded_dirs], -1)

            embedded = embedded.to(torch.float32)
            outputs_flat = self.batchify(fn, self.chunk)(embedded)
            outputs = torch.reshape(
                outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]
            )
            return outputs 