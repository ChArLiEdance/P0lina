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
        pass

    def summarize(self):
        """
        Write your codes here.
        """
        pass
