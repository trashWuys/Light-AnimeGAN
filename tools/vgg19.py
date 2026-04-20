import torch
import torch.nn as nn
import torchvision.models as models

class VGG19(nn.Module):
    """
    提取 relu3_1 和 relu4_1 特征，与原版AnimeGANv2 VGG使用方式对齐。
    输入：[-1, 1] 范围的 RGB tensor (B, 3, H, W)
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True)
        features = vgg.features

        # relu2_2: features[0:9]   → 用于style（浅层纹理）
        # relu3_1: features[0:14]  → 用于content（结构）
        # relu4_1: features[0:23]  → 用于content（语义）

        self.slice1 = nn.Sequential(*list(features)[:9])   # relu2_2 (style)
        self.slice2 = nn.Sequential(*list(features)[9:18])  # relu3_2 (style)
        self.slice3 = nn.Sequential(*list(features)[18:26]) # relu4_2 (content)

        for param in self.parameters():
            param.requires_grad = False

        # ✅ ImageNet 归一化均值和标准差（适配[-1,1]输入的反归一化）
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, x):
        """将 [-1, 1] 的 RGB 转换为 VGG19 期望的 ImageNet 归一化输入"""
        x = (x + 1.0) / 2.0          # → [0, 1]
        x = (x - self.mean) / self.std  # → ImageNet normalized
        return x

    def forward(self, x):
        x = self.preprocess(x)
        h1 = self.slice1(x)   # relu2_2 → style特征
        h2 = self.slice2(h1)  # relu3_2 → style特征
        h3 = self.slice3(h2)  # relu4_2 → content特征

        content_features = [h3]          # 内容特征列表
        style_features   = [h1, h2]      # 风格特征列表
        return content_features, style_features