import numpy as np
import torch
import torch.nn.functional as F
from src.config import cfg

def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    device = bins.device
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.from_numpy(u).to(device)
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])
    return samples

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    device = raw.device
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    ones_tensor = torch.tensor([1e10], device=device).expand(dists[..., :1].shape)
    dists = torch.cat([dists, ones_tensor], -1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    rgb = torch.sigmoid(raw[..., :3])
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape, device=device) * raw_noise_std
        if pytest:
            np.random.seed(0)
            noise = torch.from_numpy(np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std).to(device)
    alpha = raw2alpha(raw[..., 3] + noise, dists)
    ones_alpha = torch.ones((alpha.shape[0], 1), device=device)
    weights = alpha * torch.cumprod(torch.cat([ones_alpha, 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)
    eps_tensor = 1e-10 * torch.ones_like(depth_map)
    disp_map = 1./torch.max(eps_tensor, depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)
    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[..., None])
    return rgb_map, disp_map, acc_map, weights, depth_map

def batchify(fn, chunk):
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

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
        self.N_importance = cfg.task_arg.N_importance if hasattr(cfg.task_arg, 'N_importance') else 128
        self.use_viewdirs = cfg.task_arg.use_viewdirs if hasattr(cfg.task_arg, 'use_viewdirs') else True
        self.white_bkgd = cfg.task_arg.white_bkgd if hasattr(cfg.task_arg, 'white_bkgd') else True
        self.raw_noise_std = cfg.task_arg.raw_noise_std if hasattr(cfg.task_arg, 'raw_noise_std') else 0.0
        self.perturb = cfg.task_arg.perturb if hasattr(cfg.task_arg, 'perturb') else 1.0
        self.lindisp = cfg.task_arg.lindisp if hasattr(cfg.task_arg, 'lindisp') else False
        self.chunk_size = cfg.task_arg.chunk_size if hasattr(cfg.task_arg, 'chunk_size') else 4096
        #pass

    def render_rays(self, ray_batch, retraw=False, pytest=False):
        N_rays = ray_batch.shape[0]
        device = ray_batch.device
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]
        t_vals = torch.linspace(0., 1., steps=self.N_samples, device=device)
        if not self.lindisp:
            z_vals = near * (1.-t_vals) + far * (t_vals)
        else:
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
        z_vals = z_vals.expand([N_rays, self.N_samples])
        if self.perturb > 0.:
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand(z_vals.shape, device=device)
            if pytest:
                np.random.seed(0)
                t_rand = torch.from_numpy(np.random.rand(*list(z_vals.shape))).to(device)
            z_vals = lower + (upper - lower) * t_rand
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        if viewdirs is not None:
            viewdirs = viewdirs.to(device)
        raw = self.net(pts, viewdirs, model="coarse")
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, self.raw_noise_std, self.white_bkgd, pytest=pytest)
        if self.N_importance > 0:
            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], self.N_importance, det=(self.perturb == 0.), pytest=pytest)
            z_samples = z_samples.detach()
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
            raw = self.net(pts, viewdirs, model="fine")
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, self.raw_noise_std, self.white_bkgd, pytest=pytest)
        ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
        if retraw:
            ret['raw'] = raw
        if self.N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)
        return ret

    def batchify_rays(self, rays_flat, **kwargs):
        all_ret = {}
        for i in range(0, rays_flat.shape[0], self.chunk_size):
            ret = self.render_rays(rays_flat[i:i+self.chunk_size], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret

    def render(self, batch):
        """
        Main rendering function，对齐主分支，支持全图/批量自动reshape、near/far类型兼容、viewdirs归一化、分块渲染等。
        """
        rays_o = batch['rays_o']  # ray origins
        rays_d = batch['rays_d']  # ray directions
        near = batch.get('near', 2.0)
        far = batch.get('far', 6.0)
        # 处理DataLoader批量维度
        if len(rays_o.shape) == 3 and rays_o.shape[0] == 1:
            rays_o = rays_o.squeeze(0)
            rays_d = rays_d.squeeze(0)
        # near/far类型兼容
        if isinstance(near, torch.Tensor):
            if near.dim() == 1 and near.shape[0] == 1:
                near = near.item()
            elif near.dim() == 0:
                near = near.item()
        if isinstance(far, torch.Tensor):
            if far.dim() == 1 and far.shape[0] == 1:
                far = far.item()
            elif far.dim() == 0:
                far = far.item()
        # 支持全图自动reshape
        original_shape = None
        if len(rays_o.shape) == 3:
            H, W = rays_o.shape[:2]
            original_shape = (H, W)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
        # near/far广播
        if isinstance(near, (int, float)):
            near = near * torch.ones_like(rays_d[..., :1])
        if isinstance(far, (int, float)):
            far = far * torch.ones_like(rays_d[..., :1])
        rays = torch.cat([rays_o, rays_d, near, far], -1)  # (N_rays, 8)
        if self.use_viewdirs:
            viewdirs = rays_d
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            rays = torch.cat([rays, viewdirs], -1)
        all_ret = self.batchify_rays(rays)
        if original_shape is not None:
            H, W = original_shape
            for k in all_ret:
                if len(all_ret[k].shape) > 1:
                    all_ret[k] = all_ret[k].reshape(H, W, -1)
                else:
                    all_ret[k] = all_ret[k].reshape(H, W)
        return all_ret

        