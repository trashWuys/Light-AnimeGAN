import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class ConvLReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.block = nn.Sequential(
            # ✅ 核心修复 1：彻底移除 InstanceNorm，使用谱归一化 (spectral_norm)
            # 谱归一化严格限制了判别器的利普希茨常数，防止 D 远强于 G
            spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class D_net(nn.Module):
    def __init__(self, channel=64, n_dis=3):
        super().__init__()
        layers = [
            # 第一层同样加上谱归一化
            spectral_norm(nn.Conv2d(3, channel, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        in_ch = channel
        for i in range(1, n_dis):
            out_ch = in_ch * 2
            layers += [
                ConvLReLU(in_ch, out_ch, stride=2),
                ConvLReLU(out_ch, out_ch, stride=1),
            ]
            in_ch = out_ch

        layers += [
            ConvLReLU(in_ch, in_ch * 2, stride=1),
            # 最终输出层 - 无激活，输出logit，依然需要谱归一化
            spectral_norm(nn.Conv2d(in_ch * 2, 1, kernel_size=3, stride=1, padding=1)),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)