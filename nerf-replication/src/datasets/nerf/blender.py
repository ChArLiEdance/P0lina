import torch.utils.data as data
import torch
import numpy as np
import json
import os
import cv2
import imageio
from src.config import cfg

#---------------------------------------------------------------c2w转换矩阵-----------------------------------------------------------------
#沿着z轴平移矩阵
trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()
#绕y轴逆时针旋转的矩阵
rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()
#绕x轴逆时针旋转的矩阵
rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    """
    :    theta: 旋转角度，绕y轴
    :    phi: 俯仰角度，绕x轴
    :    radius: 半径
    :return: 4x4的变换矩阵
    先沿着z轴移动半径距离
    然后绕y轴逆时针旋转theta角度
    再绕x轴逆时针旋转phi角度
    最后将坐标系从右手系转换为左手系
    获得了[4*4]变换矩阵，可以从球坐标系转化为笛卡尔坐标系

    """
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

#---------------------------------------------------------------get_rays-----------------------------------------------------------------
def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='ij')
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d

#-----------------------------------------------------dataset类------------------------------------------------------------

class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        """
        Description:
            __init__ 函数负责从磁盘中 load 指定格式的文件，计算并存储为特定形式

        Input:
            @kwargs: 读取的参数
        Output:
            None
        """
        super(Dataset, self).__init__()
        print("Initializing NeRF Blender Dataset...")
        # 参数优先级：kwargs > cfg > 默认
        self.split = kwargs.get('split', 'train')
        scene = kwargs.get('scene', getattr(cfg, 'scene', 'lego'))
        if hasattr(cfg, 'train_dataset') and hasattr(cfg.train_dataset, 'data_root'):
            data_root = kwargs.get('data_root', cfg.train_dataset.data_root)
        else:
            data_root = kwargs.get('data_root', 'data/nerf_synthetic')
        if hasattr(cfg, 'train_dataset') and hasattr(cfg.train_dataset, 'input_ratio'):
            input_ratio = kwargs.get('input_ratio', cfg.train_dataset.input_ratio)
        else:
            input_ratio = kwargs.get('input_ratio', 1.0)
        self.basedir = os.path.join(data_root, scene)
        self.input_ratio = input_ratio
        self.white_bkgd = getattr(cfg.task_arg, 'white_bkgd', True) if hasattr(cfg, 'task_arg') else True
        self.N_rays = getattr(cfg.task_arg, 'N_rays', 1024) if hasattr(cfg, 'task_arg') else 1024
        self.no_batching = getattr(cfg.task_arg, 'no_batching', True) if hasattr(cfg, 'task_arg') else True
        self.test_skip = getattr(cfg.task_arg, 'test_skip', 1) if hasattr(cfg, 'task_arg') else 1
        print(f"Dataset config: split={self.split}, scene={scene}, data_root={data_root}")
        print(f"Base directory: {self.basedir}")
        if not os.path.exists(self.basedir):
            raise FileNotFoundError(f"Dataset directory not found: {self.basedir}")
        self._load_data()
        if self.split == 'train' and self.no_batching and len(self.image_paths) > 0:
            total_pixels = len(self.image_paths) * self.H * self.W
            if total_pixels <= 10000000:
                print("Preparing ray batches for training...")
                self._prepare_ray_batch()
            else:
                print(f"Too many pixels ({total_pixels}), using on-demand ray generation")
        print(f"Dataset initialization completed: {len(self.image_paths)} images")

    def _load_data(self):
        print("Loading blender dataset...")
        splits = ['train', 'val', 'test']
        metas = {}
        for s in splits:
            json_file = os.path.join(self.basedir, f'transforms_{s}.json')
            if os.path.exists(json_file):
                with open(json_file, 'r') as fp:
                    metas[s] = json.load(fp)
                print(f"Loaded {len(metas[s]['frames'])} frames for {s}")
            else:
                print(f"Warning: {json_file} not found")
        if not metas:
            raise FileNotFoundError("No transform files found")
        H, W = 800, 800
        camera_angle_x = float(metas['train']['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        if self.input_ratio != 1.0:
            H = int(H * self.input_ratio)
            W = int(W * self.input_ratio)
            focal = focal * self.input_ratio
        print(f"Target image size: {H}x{W}, focal: {focal}")
        self.K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
        self.H, self.W, self.focal = H, W, focal
        self.hwf = [H, W, focal]
        if self.split == 'train':
            meta = metas['train']
            skip = 1
        elif self.split == 'val':
            meta = metas['val']
            skip = 1
        else:
            meta = metas['test']
            skip = self.test_skip
        self.image_paths = []
        self.poses_list = []
        print(f"Processing {self.split} data with skip={skip}...")
        for i, frame in enumerate(meta['frames'][::skip]):
            img_path = os.path.join(self.basedir, frame['file_path'] + '.png')
            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                self.poses_list.append(np.array(frame['transform_matrix'], dtype=np.float32))
            else:
                print(f"Warning: Image file not found: {img_path}")
        print(f"Found {len(self.image_paths)} valid images for {self.split}")
        if len(self.image_paths) == 0:
            raise RuntimeError("No images found")
        self.poses = torch.from_numpy(np.array(self.poses_list)).float()
        self.K = torch.from_numpy(self.K).float()
        if self.split == 'train' and len(self.image_paths) <= 100:
            print("Pre-loading training images...")
            self._load_all_images()
        else:
            print(f"Using on-demand loading for {len(self.image_paths)} images")
            self.imgs = None
        self.render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40+1)[:-1]], 0)
        print(f"Dataset setup completed: {len(self.image_paths)} images of target size {H}x{W}")

    def _load_all_images(self):
        imgs = []
        print("Loading all images into memory...")
        for i, img_path in enumerate(self.image_paths):
            try:
                img = imageio.imread(img_path)
                if self.input_ratio != 1.0:
                    img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
                imgs.append(img)
                if (i + 1) % 20 == 0:
                    print(f"Loaded {i + 1}/{len(self.image_paths)} images")
            except Exception as e:
                print(f"Warning: Failed to load image {img_path}: {e}")
        if imgs:
            imgs = (np.array(imgs) / 255.).astype(np.float32)
            if self.white_bkgd:
                imgs = imgs[..., :3] * imgs[..., -1:] + (1. - imgs[..., -1:])
            else:
                imgs = imgs[..., :3]
            self.imgs = torch.from_numpy(imgs).float()
            print(f"All images loaded: {self.imgs.shape}")
        else:
            raise RuntimeError("Failed to load any images")

    def _load_single_image(self, index):
        img_path = self.image_paths[index]
        try:
            img = imageio.imread(img_path)
            if self.input_ratio != 1.0:
                img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
            img = (img / 255.).astype(np.float32)
            if self.white_bkgd:
                img = img[..., :3] * img[..., -1:] + (1. - img[..., -1:])
            else:
                img = img[..., :3]
            return torch.from_numpy(img).float()
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros(self.H, self.W, 3, dtype=torch.float32)

    def _prepare_ray_batch(self):
        print('Preparing ray batches...')
        if self.split != 'train':
            print("Skip ray precomputation for non-training split")
            return
        if len(self.imgs) == 0:
            print("No images to precompute rays")
            return
        print(f"Computing rays for {len(self.imgs)} training images of size {self.H}x{self.W}")
        max_rays_per_image = 1024
        total_rays_per_image = self.H * self.W
        if total_rays_per_image <= max_rays_per_image:
            rays_list = []
            for i in range(len(self.imgs)):
                pose = self.poses[i, :3, :4]
                rays_o, rays_d = get_rays(self.H, self.W, self.K, pose)
                img = self.imgs[i]
                rays_o_flat = rays_o.view(-1, 3)
                rays_d_flat = rays_d.view(-1, 3)
                img_flat = img.view(-1, 3)
                rays_rgb = torch.stack([rays_o_flat, rays_d_flat, img_flat], 0).transpose(0, 1)
                rays_list.append(rays_rgb)
            self.rays_rgb = torch.cat(rays_list, 0)
            indices = torch.randperm(self.rays_rgb.shape[0])
            self.rays_rgb = self.rays_rgb[indices]
            self.i_batch = 0
            print(f'Ray batches prepared: {self.rays_rgb.shape[0]} rays total')
        else:
            print(f"Images too large ({self.H}x{self.W}), will generate rays on-demand")
            self.rays_rgb = None
            self.i_batch = 0

    def __getitem__(self, index):
        """
        Description:
            __getitem__ 函数负责在运行时提供给网络一次训练需要的输入，以及 ground truth 的输出
        对 NeRF 来说，分别是 1024 条光线以及 1024 个 RGB值

        Input:
            @index: 图像下标, 范围为 [0, len-1]
        Output:
            @ret: 包含所需数据的字典
        """
        if self.split == 'train' and self.no_batching:
            if hasattr(self, 'rays_rgb') and self.rays_rgb is not None:
                batch = self.rays_rgb[self.i_batch:self.i_batch + self.N_rays]
                batch = batch.transpose(0, 1)
                batch_rays, target_s = batch[:2], batch[2]
                self.i_batch += self.N_rays
                if self.i_batch >= self.rays_rgb.shape[0]:
                    indices = torch.randperm(self.rays_rgb.shape[0])
                    self.rays_rgb = self.rays_rgb[indices]
                    self.i_batch = 0
                ret = {
                    'rays_o': batch_rays[0],
                    'rays_d': batch_rays[1],
                    'target_s': target_s,
                    'H': self.H,
                    'W': self.W,
                    'K': self.K,
                    'near': 2.0,
                    'far': 6.0
                }
            else:
                img_i = np.random.randint(0, len(self.image_paths))
                if self.imgs is not None:
                    img = self.imgs[img_i]
                else:
                    img = self._load_single_image(img_i)
                pose = self.poses[img_i, :3, :4]
                rays_o, rays_d = get_rays(self.H, self.W, self.K, pose)
                coords = torch.stack(torch.meshgrid(
                    torch.arange(self.H, dtype=torch.long), 
                    torch.arange(self.W, dtype=torch.long),
                    indexing='ij'
                ), -1)
                coords = coords.reshape(-1, 2)
                select_inds = np.random.choice(coords.shape[0], size=self.N_rays, replace=False)
                select_coords = coords[select_inds]
                rays_o_selected = rays_o[select_coords[:, 0], select_coords[:, 1]]
                rays_d_selected = rays_d[select_coords[:, 0], select_coords[:, 1]]
                target_s = img[select_coords[:, 0], select_coords[:, 1]]
                ret = {
                    'rays_o': rays_o_selected,
                    'rays_d': rays_d_selected,
                    'target_s': target_s,
                    'H': self.H,
                    'W': self.W,
                    'K': self.K,
                    'near': 2.0,
                    'far': 6.0
                }
        else:
            if self.imgs is not None:
                img = self.imgs[index]
            else:
                img = self._load_single_image(index)
            pose = self.poses[index, :3, :4]
            rays_o, rays_d = get_rays(self.H, self.W, self.K, pose)
            rays_o_flat = rays_o.view(-1, 3)
            rays_d_flat = rays_d.view(-1, 3)
            img_flat = img.view(-1, 3)
            ret = {
                'rays_o': rays_o_flat,
                'rays_d': rays_d_flat,
                'target_s': img_flat,
                'H': self.H,
                'W': self.W,
                'K': self.K,
                'pose': pose,
                'near': 2.0,
                'far': 6.0,
                'img_idx': int(index)
            }
        return ret

    def __len__(self):
        """
        Description:
            __len__ 函数返回训练或者测试的数量

        Input:
            None
        Output:
            @len: 训练或者测试的数量
        """
        if self.split == 'train' and self.no_batching and hasattr(self, 'rays_rgb') and self.rays_rgb is not None:
            return self.rays_rgb.shape[0] // self.N_rays
        else:
            return len(self.image_paths)
