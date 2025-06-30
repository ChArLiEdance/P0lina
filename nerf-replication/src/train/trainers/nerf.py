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
        pass
