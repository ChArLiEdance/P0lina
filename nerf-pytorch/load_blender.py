'''
加载训练集、验证集和测试集的图像和相机变换矩阵（pose）。

对图像进行归一化处理，并根据需要进行分辨率缩小。

计算相机的焦距和视角。

生成用于渲染的不同视角（40个渲染位置）。

返回处理后的图像、变换矩阵、渲染视角、相机参数和数据拆分索引。
'''
import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2

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


def load_blender_data(basedir, half_res=False, testskip=1):
    """

    :param basedir: 数据集所在目录
    :param half_res: 是否使用半分辨率
    :param testskip: 测试集跳过的帧数
    :return: imgs: 图像数据，形状为[N, H, W, 4]，4通道（RGBA）
             poses: 相机位姿，形状为[N, 4, 4]
             render_poses: 渲染时的相机位姿
             [H, W, focal]: 图像的高度、宽度和焦距
             i_split: 每个数据集的索引列表
    """
    #加载三个不同的数据集：train, val, test，分别是训练集、验证集和测试集

    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)
    #初始化图像和位姿列表
    # counts用于记录每个数据集的图像数量
    # all_imgs用于存储所有图像数据
    # all_poses用于存储所有相机位姿
    all_imgs = []
    all_poses = []
    counts = [0]
    # 遍历每个数据集，加载图像和位姿
    # 对于训练集和验证集，跳过的帧数为1，对于测试集，根据testskip参数跳过指定的帧数
    # 每个数据集的图像和位姿都存储在imgs和poses中
    # 最后将所有图像和位姿合并为一个大数组
    # i_split用于存储每个数据集的索引范围
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))#读取图像
            poses.append(np.array(frame['transform_matrix']))#获取相机变换矩阵
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        #将图像和位姿添加到总列表中
        all_imgs.append(imgs)
        all_poses.append(poses)

    #i_split用于存储每个数据集的索引范围
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    #将所有图像和位姿合并为一个大数组

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2] #获取图像的高度和宽度
    camera_angle_x = float(meta['camera_angle_x'])#获取相机的水平视角
    focal = .5 * W / np.tan(.5 * camera_angle_x)#计算焦距
    #生成渲染时的相机位姿
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    #处理半分辨率的情况
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    return imgs, poses, render_poses, [H, W, focal], i_split


