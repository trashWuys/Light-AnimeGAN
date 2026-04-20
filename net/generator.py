import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────
# 基础模块
# ─────────────────────────────────────────────

class ConvNormLReLU(nn.Module):
    """Conv + InstanceNorm + LeakyReLU"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1,
                 padding=1, pad_mode='reflect', groups=1, bias=False):
        super().__init__()
        layers = []
        # 反射填充更贴近原版TF的REFLECT padding
        if pad_mode == 'reflect':
            layers.append(nn.ReflectionPad2d(padding))
            padding = 0
        layers += [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride,
                      padding=padding, groups=groups, bias=bias),
            nn.InstanceNorm2d(out_ch, affine=True),  # affine=True 保留可学习参数
            nn.LeakyReLU(0.2, inplace=True),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class InvertedResBlock(nn.Module):
    """MobileNetV2风格的倒残差块，与原版AnimeGANv2一致"""
    def __init__(self, in_ch=256, out_ch=256, expand_ratio=2, stride=1):
        super().__init__()
        bottleneck_ch = round(in_ch * expand_ratio)
        self.use_res = (stride == 1 and in_ch == out_ch)

        self.block = nn.Sequential(
            # Pointwise
            ConvNormLReLU(in_ch, bottleneck_ch, kernel_size=1, padding=0, pad_mode='none'),
            # Depthwise
            ConvNormLReLU(bottleneck_ch, bottleneck_ch, kernel_size=3, stride=stride,
                          padding=1, pad_mode='reflect', groups=bottleneck_ch),
            # Pointwise linear (无激活)
            nn.Conv2d(bottleneck_ch, out_ch, kernel_size=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
        )

    def forward(self, x):
        if self.use_res:
            return x + self.block(x)
        return self.block(x)

# ─────────────────────────────────────────────
# 生成器网络 G_net
# ─────────────────────────────────────────────

class G_net(nn.Module):
    def __init__(self):
        super().__init__()

        self.encode = nn.Sequential(
            # Block A: 下采样 ×1
            ConvNormLReLU(3,   32, kernel_size=7, padding=3),
            ConvNormLReLU(32,  64, stride=2, padding=1),
            ConvNormLReLU(64,  64, padding=1),

            # Block B: 下采样 ×2
            ConvNormLReLU(64,  128, stride=2, padding=1),
            ConvNormLReLU(128, 128, padding=1),
        )

        self.res_blocks = nn.Sequential(
            InvertedResBlock(128, 256, expand_ratio=2),
            InvertedResBlock(256, 256, expand_ratio=2),
            InvertedResBlock(256, 256, expand_ratio=2),
            InvertedResBlock(256, 256, expand_ratio=2),
            InvertedResBlock(256, 128, expand_ratio=2),  # 最后一个块压缩回128
        )

        self.decode = nn.Sequential(
            # 上采样 ×2
            ConvNormLReLU(128, 128, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvNormLReLU(128, 64, padding=1),

            # 上采样 ×2
            ConvNormLReLU(64,  64, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvNormLReLU(64,  32, padding=1),

            # 最终输出层
            nn.ReflectionPad2d(3),
            nn.Conv2d(32, 3, kernel_size=7, bias=False),
            # ✅ 关键修复：必须使用 Tanh，保证输出在 [-1, 1]
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.res_blocks(x)
        x = self.decode(x)
        return x