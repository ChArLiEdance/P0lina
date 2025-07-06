import numpy as np
from src.config import cfg
import os
import torch.nn.functional as F
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import json
import warnings
import torch

warnings.filterwarnings("ignore", category=UserWarning)

def to8b(x):
    return (255*np.clip(x, 0, 1)).astype(np.uint8)

class Evaluator:
    def __init__(self):
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.imgs = []
        self.img_names = []

    def psnr_metric(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt) ** 2)
        if mse == 0:
            return float('inf')
        psnr = -10 * np.log10(mse)
        return psnr

    def ssim_metric(self, img_pred, img_gt, batch, id, num_imgs):
        # 修复：确保 id 是 Python 整数
        if isinstance(id, torch.Tensor):
            if id.dim() == 0:
                id = id.item()
            else:
                id = id.flatten()[0].item()
        id = int(id)
        result_dir = os.path.join(cfg.result_dir, "images")
        os.makedirs(result_dir, exist_ok=True)
        if isinstance(img_pred, torch.Tensor):
            img_pred = img_pred.detach().cpu().numpy()
        if isinstance(img_gt, torch.Tensor):
            img_gt = img_gt.detach().cpu().numpy()
        if img_pred.shape != img_gt.shape:
            raise ValueError(f"Shape mismatch in SSIM: pred={img_pred.shape}, gt={img_gt.shape}")
        img_pred = np.clip(img_pred, 0, 1)
        img_gt = np.clip(img_gt, 0, 1)
        cv2.imwrite(
            "{}/view{:03d}_pred.png".format(result_dir, id),
            to8b(img_pred[..., [2, 1, 0]])
        )
        cv2.imwrite(
            "{}/view{:03d}_gt.png".format(result_dir, id),
            to8b(img_gt[..., [2, 1, 0]])
        )
        img_pred_uint8 = to8b(img_pred)
        img_gt_uint8 = to8b(img_gt)
        if img_pred_uint8.ndim == 3:
            ssim_val = compare_ssim(
                img_pred_uint8, img_gt_uint8, 
                data_range=255, 
                multichannel=True,
                channel_axis=-1
            )
        else:
            ssim_val = compare_ssim(
                img_pred_uint8, img_gt_uint8, 
                data_range=255
            )
        return ssim_val

    def evaluate(self, output, batch):
        # Extract predictions and ground truth
        if 'rgb_map' in output:
            img_pred = output['rgb_map']
        elif 'rgb_pred' in output:
            img_pred = output['rgb_pred']
        else:
            raise KeyError("No RGB prediction found in output")
        if 'target_s' in batch:
            img_gt = batch['target_s']
        else:
            raise KeyError("No ground truth target found in batch")
        if isinstance(img_pred, torch.Tensor):
            img_pred = img_pred.detach().cpu().numpy()
        if isinstance(img_gt, torch.Tensor):
            img_gt = img_gt.detach().cpu().numpy()
        H = batch.get('H', 400)
        W = batch.get('W', 400)
        if img_pred.ndim == 3 and img_pred.shape[0] == H and img_pred.shape[1] == W:
            target_shape = (H, W, 3)
        elif img_pred.ndim == 2 and img_pred.shape[0] == H * W:
            img_pred = img_pred.reshape(H, W, 3)
            target_shape = (H, W, 3)
        else:
            if img_pred.size == H * W * 3:
                img_pred = img_pred.reshape(H, W, 3)
                target_shape = (H, W, 3)
            else:
                raise ValueError(f"Cannot reshape img_pred with shape {img_pred.shape} to ({H}, {W}, 3)")
        if img_gt.shape != target_shape:
            if img_gt.size == H * W * 3:
                img_gt = img_gt.reshape(H, W, 3)
            else:
                raise ValueError(f"Cannot reshape img_gt with shape {img_gt.shape} to {target_shape}")
        img_pred = np.clip(img_pred, 0, 1)
        img_gt = np.clip(img_gt, 0, 1)
        mse = np.mean((img_pred - img_gt) ** 2)
        psnr = self.psnr_metric(img_pred, img_gt)
        img_idx = batch.get('img_idx', len(self.imgs))
        if isinstance(img_idx, torch.Tensor):
            if img_idx.dim() == 0:
                img_idx = img_idx.item()
            else:
                img_idx = img_idx.flatten()[0].item()
        img_idx = int(img_idx)
        ssim_val = self.ssim_metric(img_pred, img_gt, batch, img_idx, len(self.imgs))
        self.mse.append(mse)
        self.psnr.append(psnr)
        self.ssim.append(ssim_val)
        self.imgs.append(img_pred)
        self.img_names.append(f"view{img_idx:03d}")
        print(f"Image {img_idx:03d}: MSE={mse:.6f}, PSNR={psnr:.2f}, SSIM={ssim_val:.4f}")
        return {
            'mse': mse,
            'psnr': psnr,
            'ssim': ssim_val
        }

    def summarize(self):
        if len(self.mse) == 0:
            print("No evaluation results to summarize")
            return {}
        mean_mse = np.mean(self.mse)
        mean_psnr = np.mean(self.psnr)
        mean_ssim = np.mean(self.ssim)
        std_mse = np.std(self.mse)
        std_psnr = np.std(self.psnr)
        std_ssim = np.std(self.ssim)
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Number of images: {len(self.mse)}")
        print(f"MSE:  {mean_mse:.6f} ± {std_mse:.6f}")
        print(f"PSNR: {mean_psnr:.2f} ± {std_psnr:.2f} dB")
        print(f"SSIM: {mean_ssim:.4f} ± {std_ssim:.4f}")
        print("="*50)
        results = {
            'summary': {
                'mean_mse': float(mean_mse),
                'mean_psnr': float(mean_psnr),
                'mean_ssim': float(mean_ssim),
                'std_mse': float(std_mse),
                'std_psnr': float(std_psnr),
                'std_ssim': float(std_ssim),
                'num_images': len(self.mse)
            },
            'per_image': []
        }
        for i in range(len(self.mse)):
            results['per_image'].append({
                'image_name': self.img_names[i],
                'mse': float(self.mse[i]),
                'psnr': float(self.psnr[i]),
                'ssim': float(self.ssim[i])
            })
        result_file = os.path.join(cfg.result_dir, "evaluation_results.json")
        os.makedirs(cfg.result_dir, exist_ok=True)
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to: {result_file}")
        return results['summary']
    
    def reset(self):
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.imgs = []
        self.img_names = []
