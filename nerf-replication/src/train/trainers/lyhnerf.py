import torch
import torch.nn as nn
from src.models.nerf.renderer.volume_renderer import Renderer


class NetworkWrapper(nn.Module):
    def __init__(self, net, train_loader):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.renderer = Renderer(self.net)

        # add metrics here

    def forward(self, batch):
        """
        Write your codes here.
        """

        #Render the batch using the renderer
        render_output = self.renderer.render(batch)
        # rendered_outputs is a dictionary containing rgb, depth, acc, etc.

        #compare the rendered outputs with the ground truth
        pred_rgb= render_output['rgb_map']
        gt_rgb= batch['rgb']

        #计算loss
        loss = nn.functional.mse_loss(pred_rgb, gt_rgb)

        loss_stats={
            'loss': loss.detach(),
            'mse': loss.detach()

        }
        
        #detach()用于从计算图中分离张量，避免梯度计算

        # 计算图像统计信息（用于可视化）
        image_stats = {
            'pred_rgb': pred_rgb.detach(),
            'gt_rgb': gt_rgb.detach(),
            'rgb_map': render_output['rgb_map'].detach(),
            'disp_map': render_output['disp_map'].detach(),
            'acc_map': render_output['acc_map'].detach()
        }
        return render_output, loss, loss_stats, image_stats
        
        #pass
