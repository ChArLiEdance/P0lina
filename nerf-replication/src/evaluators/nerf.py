import numpy as np
from src.config import cfg
import os
import torch.nn.functional as F
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import json
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class Evaluator:
    def __init__(
        self,
    ):
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.imgs = []

    def psnr_metric(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt) ** 2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def ssim_metric(self, img_pred, img_gt, batch, id, num_imgs):
        result_dir = os.path.join(cfg.result_dir, "images")
        os.system("mkdir -p {}".format(result_dir))
        cv2.imwrite(
            "{}/view{:03d}_pred.png".format(result_dir, id),
            (img_pred[..., [2, 1, 0]] * 255),
        )
        cv2.imwrite(
            "{}/view{:03d}_gt.png".format(result_dir, id),
            (img_gt[..., [2, 1, 0]] * 255),
        )
        img_pred = (img_pred * 255).astype(np.uint8)

        ssim = compare_ssim(img_pred, img_gt, win_size=101, full=True)
        return ssim

    def evaluate(self, output, batch):
        """
        Write your codes here.
        """
        #calculate psnr and ssim
        pred_rgb = output["rgb"]
        gt_rgb = batch["rgb"]

        # 确保数据在CPU上并转换为numpy
        pred_rgb = pred_rgb.detach().cpu().numpy()
        gt_rgb = gt_rgb.detach().cpu().numpy()

        # 计算MSE
        mse = np.mean((pred_rgb - gt_rgb) ** 2)
        self.mse.append(mse)

        # 计算PSNR
        psnr_val = self.psnr_metric(pred_rgb, gt_rgb)
        self.psnr.append(psnr_val)

        # 计算SSIM（需要重建完整图像）
        # 获取当前批次的索引信息
        batch_idx = getattr(batch, 'batch_idx', len(self.imgs))

        # 将当前批次的预测结果存储
        self.imgs.append({
            'pred': pred_rgb,
            'gt': gt_rgb,
            'batch_idx': batch_idx
        })

        # 如果累积了足够的批次来重建完整图像，计算SSIM
        if len(self.imgs) >= (self.H * self.W) // self.N_rays:
            # 重建完整图像
            pred_full = self.reconstruct_full_image(self.imgs, 'pred')
            gt_full = self.reconstruct_full_image(self.imgs, 'gt')
            
            if pred_full is not None and gt_full is not None:
                ssim_val = self.ssim_metric(pred_full, gt_full, batch, batch_idx, len(self.imgs))
                self.ssim.append(ssim_val)
            
            # 清空图像缓存
            self.imgs = []
        
        return None
        #pass
    def reconstruct_full_image(self, img_batches, key):
        """
        从多个批次重建完整图像
        
        Args:
            img_batches: 图像批次列表
            key: 'pred' 或 'gt'
        
        Returns:
            重建的完整图像 [H, W, 3] 或 None
        """
        try:
            # 计算完整图像需要的像素数
            total_pixels = self.H * self.W
            current_pixels = sum(len(batch[key]) for batch in img_batches)
            
            if current_pixels >= total_pixels:
                # 重建完整图像
                full_img = np.zeros((self.H, self.W, 3), dtype=np.float32)
                pixel_count = 0
                
                for batch in img_batches:
                    batch_pixels = len(batch[key])
                    if pixel_count + batch_pixels <= total_pixels:
                        # 将批次像素填充到完整图像中
                        start_idx = pixel_count
                        end_idx = pixel_count + batch_pixels
                        
                        # 计算在完整图像中的位置
                        start_h = start_idx // self.W
                        start_w = start_idx % self.W
                        end_h = end_idx // self.W
                        end_w = end_idx % self.W
                        
                        # 填充像素
                        if start_h == end_h:
                            # 同一行
                            full_img[start_h, start_w:end_w] = batch[key][:end_w-start_w]
                        else:
                            # 跨行
                            # 第一行
                            first_row_pixels = self.W - start_w
                            full_img[start_h, start_w:] = batch[key][:first_row_pixels]
                            
                            # 中间完整行
                            for h in range(start_h + 1, end_h):
                                row_start = first_row_pixels + (h - start_h - 1) * self.W
                                row_end = row_start + self.W
                                full_img[h, :] = batch[key][row_start:row_end]
                            
                            # 最后一行
                            last_row_pixels = end_w
                            last_row_start = first_row_pixels + (end_h - start_h - 1) * self.W
                            full_img[end_h, :last_row_pixels] = batch[key][last_row_start:]
                        
                        pixel_count += batch_pixels
                
                return full_img
            
        except Exception as e:
            print(f"重建图像时出错: {e}")
            return None
        
        return None


    def summarize(self):
        """
        汇总所有评估指标并保存结果
        
        Returns:
            包含所有指标的字典
        """

        """
        Write your codes here.
        """

        ret = {}
        
        # 计算平均指标
        if len(self.mse) > 0:
            ret['mse'] = np.mean(self.mse)
            ret['psnr'] = np.mean(self.psnr)
        
        if len(self.ssim) > 0:
            ret['ssim'] = np.mean(self.ssim)
        
        # 打印结果
        print("=" * 50)
        print("评估结果:")
        for key, value in ret.items():
            print(f"{key.upper()}: {value:.4f}")
        print("=" * 50)

        # 保存结果到JSON文件
        result_file = os.path.join(cfg.result_dir, "metrics.json")
        try:
            import json
            with open(result_file, 'w') as f:
                json.dump(ret, f, indent=2)
            print(f"指标已保存到: {result_file}")
        except Exception as e:
            print(f"保存指标文件时出错: {e}")
        
        # 清空缓存
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.imgs = []
        
        return ret
        #pass
