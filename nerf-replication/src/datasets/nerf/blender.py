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
        """
        Write your codes here.
        """


        # 从kwargs读取参数，兼容yaml配置 , kwargs是从yaml配置文件中读取的参数，
        #子lego中有split，data_root, H, W等参数
        
        self.split= kwargs.get('split', 'train')
        self.data_root=kwargs.get('data_root',getattr(cfg,'data_root',None))
        self.H = int(kwargs.get('H', 800))
        self.W = int(kwargs.get('W', 800))
        self.img_wh = (self.W, self.H)
        self.input_ratio = float(kwargs.get('input_ratio', 1.0))
        self.cams = kwargs.get('cams', None)
        self.half_res = kwargs.get('half_res', False)
        self.testskip = kwargs.get('testskip', 1)
        self.near = 2.0
        self.far = 6.0
        
        # 读取 transforms_xxx.json
        json_path = os.path.join(self.data_root, f"transforms_{self.split}.json")
        with open(json_path, 'r') as f:
            meta = json.load(f)

        # 读取图像路径和相机位姿, 
        self.image_paths = []
        self.poses = []
        self.all_rgbs = []
        self.focal = None
        self.H_ori, self.W_ori = None, None
        frames=meta['frames']


        #test 可以跳过一些样本
        if self.split == 'train' or self.testskip == 0:
            skip = 1
        else:
            skip = self.testskip
        frames = frames[::skip]


        # 支持相机选择
        if self.cams is not None and isinstance(self.cams, list):
            selected_frames = []
            for idx in self.cams:
                if idx >= 0 and idx < len(frames):
                    selected_frames.append(frames[idx])
            if selected_frames:
                frames = selected_frames
        

        img = []
        for frame in frames:
            #获取图像路径
            img_path = os.path.join(self.data_root, frame['file_path'] + '.png')
            if not os.path.exists(img_path):
                img_path = os.path.join(self.data_root, frame['file_path'] + '.jpg')
            self.image_paths.append(img_path)
            # 获取图像的原始分辨率
            self.poses.append(np.array(frame['transform_matrix'], dtype=np.float32))
            # 获取相机位姿
            img.append(imageio.imread(img_path))  # 读取图像
        img= (np.array(img) / 255.).astype(np.float32)  # 保留所有4个通道（RGBA）
        self.all_rgbs.append(img)


        # 将所有图像和位姿合并为一个大数组
        self.all_rgbs = np.concatenate(self.all_rgbs, 0)
        if self.all_rgbs.shape[-1] == 4:
            self.all_rgbs = self.all_rgbs[..., :3]
        self.pose=np.concatenate(self.poses, 0).astype(np.float32)
        self.H_ori, self.W_ori = self.all_rgbs.shape[1], self.all_rgbs.shape[2]


        # 处理输入分辨率, 如果输入分辨率小于1.0，则缩放图像
        if self.input_ratio < 1.0:
            new_H = int(self.H_ori * self.input_ratio)
            new_W = int(self.W_ori * self.input_ratio)
            imgs_resized = np.zeros((self.all_rgbs.shape[0], new_H, new_W, 3), dtype=np.float32)
            for i, img in enumerate(self.all_rgbs):
                # if img.shape[-1] == 4:
                #     img = img[..., :3]
                imgs_resized[i] = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_AREA)
                
            self.all_rgbs = imgs_resized
            self.H, self.W = new_H, new_W
        else:
            self.H, self.W = self.H_ori, self.W_ori
        
        
        # 相机内参 - 先基于原始分辨率计算焦距
        if 'fl_x' in meta:
            self.focal = meta['fl_x']
        elif 'camera_angle_x' in meta:
            # 由视场角和原始宽度计算focal
            self.focal = 0.5 * self.W_ori / np.tan(0.5 * meta['camera_angle_x'])
        else:
            raise ValueError('No focal length info in transforms json!')
        
        
        # 根据下采样比例调整焦距
        if self.input_ratio < 1.0:
            self.focal = self.focal * self.input_ratio
        if self.half_res:
            self.focal = self.focal / 2.


        # 构造相机内参矩阵K
        K = np.array([
            [self.focal, 0, 0.5 * self.W],
            [0, self.focal, 0.5 * self.H],
            [0, 0, 1]
        ], dtype=np.float32)


        # 将相机内参转换为torch.Tensor
        all_rays= []
        for pose in self.poses:
            rays_o, rays_d = get_rays_np(self.H, self.W, K, pose)
            all_rays.append(np.stack([rays_o, rays_d], axis=1).reshape(-1, 2, 3))
        self.rays = np.concatenate(all_rays, axis=0)  # [N*H*W, 2, 3]


        # flatten
        self.rays = self.rays.reshape(-1, 2, 3)  # [N*H*W, 2, 3]
        self.all_rgbs = self.all_rgbs.reshape(-1, 3)  # [N*H*W, 3]
        self.num_rays = self.rays.shape[0]


        #pass

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
        """
        Write your codes here.
        """

        
        # 获取光线和对应的 RGB 值
        N_rays= getattr(cfg, 'N_rays', 1024)
        if hasattr(self, 'N_rays'):
            N_rays = self.N_rays


        #训练时随机采样光线，测试时不随机采样
        if self.split == 'train':
            select_idx= np.random.choice(self.num_rays, N_rays, replace=False)
        else:
            start= index * N_rays
            end = min((index + 1) * N_rays, self.num_rays)
            select_idx = np.arange(start, end)

        # 获取对应的光线和 RGB 值
        rays=self.rays[select_idx]
        rgbs = self.all_rgbs[select_idx]
        rays_o =rays[:,0]  # [N_rays, 3]
        rays_d = rays[:, 1]  # [N_rays, 3]

        # 将光线和 RGB 值转换为 torch.Tensor
        # 这里的 rays_o 和 rays_d 分别是光线的起点和方向向量
        # rgbs 是对应的 RGB 颜色值
        # 返回一个字典，包含光线起点、方向和 RGB 值
        return {
            "rays_o": torch.from_numpy(rays_o).float(),
            "rays_d": torch.from_numpy(rays_d).float(),
            "rgb": torch.from_numpy(rgbs).float(),
            "near": torch.full((N_rays, 1), self.near, dtype=torch.float32),  
            "far": torch.full((N_rays, 1), self.far, dtype=torch.float32),    
        }
        #pass

    def __len__(self):
        """
        Description:
            __len__ 函数返回训练或者测试的数量

        Input:
            None
        Output:
            @len: 训练或者测试的数量
        """
        """
        Write your codes here.
        """
        
        N_rays = getattr(cfg, 'N_rays', 1024)
        if hasattr(self, 'N_rays'):
            N_rays = self.N_rays
        if self.split == 'train':
            # 训练时每次采样N_rays条，返回可采样的batch数
            return max(self.num_rays // N_rays, 1)
        else:
            # 测试时顺序采样，最后一个batch可能不足N_rays
            return (self.num_rays + N_rays - 1) // N_rays
        #pass
