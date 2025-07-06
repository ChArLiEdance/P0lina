import numpy as np
import torch
from src.config import cfg
import torch.nn.functional as F

class Renderer:
    def __init__(self, net):
        """
        初始化体渲染器，读取配置参数。
        """

        """
        Write your codes here.
        """
        self.net = net
        #读取渲染相关参数
        self.N_samples = cfg.task_arg.N_samples if hasattr(cfg.task_arg, 'N_samples') else 64
        self.N_importance = cfg.task_arg.N_importance if hasattr(cfg.task_arg, 'N_importance') else 0
        self.perturb = cfg.task_arg.perturb if hasattr(cfg.task_arg, 'perturb') else 1.0
        self.white_bkgd = cfg.task_arg.white_bkgd if hasattr(cfg.task_arg, 'white_bkgd') else False
        self.raw_noise_std = cfg.task_arg.raw_noise_std if hasattr(cfg.task_arg, 'raw_noise_std') else 0.0
        #pass

    def render(self, batch):
        
        """
        实现NeRF体渲染流程，输入batch，输出rgb、深度、透明度等。
        batch: dict，需包含rays_o, rays_d, near, far（shape: [N_rays, 3/1]）
        """
        """
        Write your codes here.
        """
        # 1. 取出射线参数
        rays_o = batch['rays_o']  # [N_rays, 3]
        rays_d = batch['rays_d']  # [N_rays, 3]
        near = batch['near']      # [N_rays, 1]
        far = batch['far']        # [N_rays, 1]
        
        # # 确保维度正确，去除可能的batch维度
        # print(f"Debug - rays_o shape: {rays_o.shape}")
        # print(f"Debug - rays_d shape: {rays_d.shape}")
        # print(f"Debug - near shape: {near.shape}")
        # print(f"Debug - far shape: {far.shape}")


        # 去除batch维度（如果存在）
        if rays_o.dim() == 3:
            rays_o = rays_o.squeeze(0)  # [1, 1024, 3] -> [1024, 3]
        if rays_d.dim() == 3:
            rays_d = rays_d.squeeze(0)  # [1, 1024, 3] -> [1024, 3]
        if near.dim() == 3:
            near = near.squeeze(0)      # [1, 1024, 1] -> [1024, 1]
        if far.dim() == 3:
            far = far.squeeze(0)        # [1, 1024, 1] -> [1024, 1]


        N_rays = rays_o.shape[0]
        device = rays_o.device

        # 2. 采样z_vals
        t_vals = torch.linspace(0., 1., steps=self.N_samples, device=device)
        z_vals = near * (1. - t_vals) + far * t_vals  # [N_rays, N_samples]
        #z_vals = z_vals.expand([N_rays, self.N_samples])

        #分层随机扰动
        #将每个相邻采样点区间 [z_i, z_{i+1}] 视为一层
        #，然后在层内做均匀随机采样，使得采样点更有随机性，有助于防止条带狀伪影。
        if self.perturb > 0:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand(z_vals.shape, device=device)
            z_vals = lower + (upper - lower) * t_rand
        
        #构造三维空间采样点
        #将每条射线的起点 o 加上方向 d 乘以各自的深度 z，得到实际的 3D 坐标
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
        
        
        #扁平化
        pts_flat = pts.reshape(-1, 3)

        #net forward
        if hasattr(self.net, 'use_viewdirs') and self.net.use_viewdirs:
            # 如果有视角信息，需要扩展视角方向
            viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)  # 归一化方向向量 [N_rays, 3]
            
            # 将pts_flat重新reshape为Network期望的形状 [N_rays, N_samples, 3]
            pts_reshaped = pts_flat.reshape(N_rays, self.N_samples, 3)
            
            # 使用Network的forward方法，传入正确形状的位置和视角信息
            raw = self.net.forward(pts_reshaped, viewdirs).reshape(N_rays, self.N_samples, -1)
        else:
           # 如果没有视角信息，将pts_flat重新reshape为Network期望的形状
            raw = self.net.forward(pts).reshape(N_rays, self.N_samples, -1)  # [N_rays, N_samples, 4]


        #体渲染积分
        rgb_map, disp_map, acc_map = self.raw2outputs(
            raw, z_vals, rays_d,
            raw_noise_std=self.raw_noise_std,
            white_bkgd=self.white_bkgd
        )

        return {
            'rgb_map': rgb_map,   # [N_rays, 3]
            'disp_map': disp_map, # [N_rays]
            'acc_map': acc_map    # [N_rays]
        }

        #pass
    @staticmethod
    def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
        """
        体渲染积分，将网络输出转为rgb、深度、透明度。
        raw：网络输出的原始数据，通常包含 RGB 颜色和透明度信息（如 [N_rays, N_samples, 4] 的张量，前三个通道是 RGB，最后一个通道是透明度）。

        z_vals：每条光线的深度值，用来计算采样点之间的距离。

        rays_d：每条光线的方向，用来计算每个采样点的距离。

        raw_noise_std：噪声标准差，添加到透明度通道用于防止数值抖动。

        white_bkgd：是否使用白色背景。

        raw2alpha：这是一个函数，用来根据 raw 中的透明度通道计算每个采样点的透明度
        """
        device = raw.device
        #透明度函数
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

        dists = z_vals[..., 1:] - z_vals[..., :-1]#计算每个采样点之间的距离
        dists = torch.cat([dists, torch.tensor([1e10], device=device).expand(dists[..., :1].shape)], -1)
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)#计算光线长度

        rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[..., 3].shape, device=device) * raw_noise_std
        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

        #计算权重
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)
        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])
        return rgb_map, disp_map, acc_map
