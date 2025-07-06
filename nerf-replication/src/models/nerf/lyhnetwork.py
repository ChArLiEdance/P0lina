import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from src.models.encoding import get_encoder
from src.config import cfg


class NeRF(nn.Module):
    def __init__(
        self, D=8, W=256, input_ch=3, input_ch_views=3, skips=[4], use_viewdirs=False
    ):
        super(NeRF, self).__init__()
        """
        D：网络的深度，表示网络的层数。

        W：网络每层的宽度，表示每层的神经元数目。

        input_ch 和 input_ch_views：分别表示输入的点坐标（x, y, z）和视角信息的维度（通常是方向向量的维度）。

        skips：在这些层之间跳跃连接（skip connections）。

        use_viewdirs：是否使用视角信息来影响输出（通常是为了改善渲染效果）。

        """

        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.output_ch = 5 if self.use_viewdirs else 4

        """
        第一个全连接层：nn.Linear(self.input_ch, self.W)，输入维度为 input_ch
        （即输入点的特征维度），输出维度为 W（即每层的神经元数量）。

        接下来的层：nn.Linear(self.W, self.W)，这些层的作用是通过 ReLU 激活函数进行特征提取。
        每一层的输入维度和输出维度都是 W，除了在某些跳跃连接层（skip connections）中
        ，输入会拼接上原始点坐标 input_pts。跳跃连接是为了防止梯度消失，帮助模型更好地学习特征。

        跳跃连接：当 i 在 self.skips 中时，输入会通过拼接操作添加原始点坐标 
        input_pts，即：nn.Linear(self.W + self.input_ch, self.W)。
        """
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, self.W)]
            + [
                (
                    nn.Linear(self.W, self.W)
                    if i not in self.skips
                    else nn.Linear(self.W + self.input_ch, self.W)
                )
                for i in range(self.D - 1)
            ]
        )
        """
        nn.Linear(self.input_ch_views + self.W, self.W // 2)：该层的输入维度是 input_ch_views + W
        （即视角信息的维度与点的特征维度相加），输出维度是 W // 2。这层会将点特征与视角特征结合，进一步提取特征。
        """

        self.views_linears = nn.ModuleList(
            [nn.Linear(self.input_ch_views + self.W, self.W // 2)]
        )
        


        #当使用视角信息时，use_viewdirs=True，网络会有额外的层来处理视角信息并生成最终的颜色和透明度。
        if self.use_viewdirs:
            # feature vector(256)
            self.feature_linear = nn.Linear(self.W, self.W)
            # alpha(1)
            self.alpha_linear = nn.Linear(self.W, 1)
            # rgb color(3)
            self.rgb_linear = nn.Linear(self.W // 2, 3)
        else:
            # output channel(default: 4)
            self.output_linear = nn.Linear(self.W, self.output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(
            x, [self.input_ch, self.input_ch_views], dim=-1
        )
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            # Apply the linear layer

            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs



    #这段代码的功能是 从 Keras 模型加载权重到 PyTorch 模型 中。
    #它假设已经训练了一个 Keras 模型，并将其权重存储在 weights 数组中。然后，
    #将这些权重加载到 PyTorch 中的相应层（如全连接层和偏置项）

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(
                np.transpose(weights[idx_pts_linears])
            )
            self.pts_linears[i].bias.data = torch.from_numpy(
                np.transpose(weights[idx_pts_linears + 1])
            )

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_feature_linear])
        )
        self.feature_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_feature_linear + 1])
        )

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(
            np.transpose(weights[idx_views_linears])
        )
        self.views_linears[0].bias.data = torch.from_numpy(
            np.transpose(weights[idx_views_linears + 1])
        )

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_rbg_linear])
        )
        self.rgb_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_rbg_linear + 1])
        )

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_alpha_linear])
        )
        self.alpha_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_alpha_linear + 1])
        )


class Network(nn.Module):
    def __init__(self):
        """
        self.N_samples：每条光线的采样点数，决定了每条射线被分割成多少个点进行渲染。

        self.N_importance：表示精细模型中额外采样的点数（用于细化渲染结果）。

        self.chunk：指定每次计算的批次大小，分割输入数据以避免内存溢出。

        self.batch_size：每批次的光线数，通常为每次并行处理的射线数量。

        self.white_bkgd：是否使用白色背景，这通常影响合成图像的背景色。

        self.use_viewdirs：是否使用视角信息来影响渲染（例如：考虑观察角度对光线颜色的影响）。

        self.device：用于指定计算设备，自动选择 GPU（如果可用），否则使用 CPU。
        """
        super(Network, self).__init__()
        self.N_samples = cfg.task_arg.N_samples
        self.N_importance = cfg.task_arg.N_importance
        self.chunk = cfg.task_arg.chunk_size
        self.batch_size = cfg.task_arg.N_rays
        self.white_bkgd = cfg.task_arg.white_bkgd
        self.use_viewdirs = cfg.task_arg.use_viewdirs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # encoder
        self.embed_fn, self.input_ch = get_encoder(cfg.network.xyz_encoder)
        self.embeddirs_fn, self.input_ch_views = get_encoder(cfg.network.dir_encoder)

        # coarse model
        self.model = NeRF(
            D=cfg.network.nerf.D,
            W=cfg.network.nerf.W,
            input_ch=self.input_ch,
            input_ch_views=self.input_ch_views,
            skips=cfg.network.nerf.skips,
            use_viewdirs=self.use_viewdirs,
        )

        # fine model
        self.model_fine = NeRF(
            D=cfg.network.nerf.D,
            W=cfg.network.nerf.W,
            input_ch=self.input_ch,
            input_ch_views=self.input_ch_views,
            skips=cfg.network.nerf.skips,
            use_viewdirs=self.use_viewdirs,
        )

        # 新增：多卡并行
        if torch.cuda.device_count() > 1:
            print("使用", torch.cuda.device_count(), "个GPU进行并行训练")
            self.model = nn.DataParallel(self.model)
            self.model_fine = nn.DataParallel(self.model_fine)

        self.model = self.model.to(self.device)
        self.model_fine = self.model_fine.to(self.device)

    def batchify(self, fn, chunk):
        """Constructs a version of 'fn' that applies to smaller batches."""

        def ret(inputs):
            return torch.cat(
                [fn(inputs[i : i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0
            )

        return ret

    def forward(self, inputs, viewdirs, model=""):
        """Prepares inputs and applies network 'fn'."""
        if model == "fine":
            fn = self.model_fine
        else:
            fn = self.model

        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        embedded = self.embed_fn(inputs_flat)

        if self.use_viewdirs:
            input_dirs = viewdirs[:, None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embedded_dirs = self.embeddirs_fn(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        embedded = embedded.to(torch.float32)
        outputs_flat = self.batchify(fn, self.chunk)(embedded)
        outputs = torch.reshape(
            outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]
        )
        return outputs
