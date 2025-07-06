import torch
import torch.nn as nn
from src.models.nerf.renderer.volume_renderer import Renderer
from src.config import cfg

# Utility functions
def img2mse(x, y):
    return torch.mean((x - y) ** 2)

def mse2psnr(x):
    return -10. * torch.log(x) / torch.log(torch.tensor([10.], device=x.device))

class NetworkWrapper(nn.Module):
    def __init__(self, net, train_loader):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.renderer = Renderer(self.net)
        self.train_loader = train_loader
        self.loss_stats = []
        self.psnr_stats = []

    def forward(self, batch):
        """
        Forward pass for training/validation
        """
        # Get renderer output
        ret = self.renderer.render(batch)
        # Extract RGB predictions and ground truth
        rgb_pred = ret['rgb_map']  # Predicted RGB values
        rgb_gt = batch['target_s']  # Ground truth RGB values
        # Compute loss
        img_loss = img2mse(rgb_pred, rgb_gt)
        loss = img_loss
        # Compute PSNR
        psnr = mse2psnr(img_loss)
        # If using hierarchical sampling (fine network)
        if 'rgb0' in ret:
            img_loss0 = img2mse(ret['rgb0'], rgb_gt)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)
            ret['psnr0'] = psnr0
            ret['loss0'] = img_loss0
        # Store outputs
        ret['loss'] = loss
        ret['img_loss'] = img_loss
        ret['psnr'] = psnr
        ret['rgb_pred'] = rgb_pred
        ret['rgb_gt'] = rgb_gt
        # Update stats for logging
        self.loss_stats.append(loss.item())
        self.psnr_stats.append(psnr.item())
        #return ret
        
        # 返回4个值以兼容现有的trainer.py
        loss_stats = {'loss': loss.detach(), 'psnr': psnr.detach()}
        image_stats = {'rgb_pred': rgb_pred.detach(), 'rgb_gt': rgb_gt.detach()}
        
        return ret, loss, loss_stats, image_stats

        
    def get_loss_stats(self):
        """Get loss statistics for logging"""
        if len(self.loss_stats) == 0:
            return {'loss': 0, 'psnr': 0}
        stats = {
            'loss': sum(self.loss_stats) / len(self.loss_stats),
            'psnr': sum(self.psnr_stats) / len(self.psnr_stats)
        }
        # Clear stats
        self.loss_stats = []
        self.psnr_stats = []
        return stats
