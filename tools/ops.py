import torch
import torch.nn as nn
import torch.nn.functional as F


def L1_loss(x, y):
    return F.l1_loss(x, y)


def Huber_loss(x, y):
    return F.smooth_l1_loss(x, y)


def content_loss(feat_real, feat_fake):
    """内容损失：VGG特征空间L1距离"""
    return F.l1_loss(feat_real, feat_fake)


def style_loss(feat_real, feat_fake):
    """风格损失：Gram矩阵L1距离"""

    def gram_matrix(feat):
        B, C, H, W = feat.shape
        f = feat.view(B, C, H * W)
        G = torch.bmm(f, f.transpose(1, 2))
        return G / (C * H * W)

    return F.l1_loss(gram_matrix(feat_real), gram_matrix(feat_fake))


def rgb2yuv(rgb):
    """
    RGB 转 YUV
    输入范围要求：[-1, 1]（由生成器 Tanh 输出保证）
    """
    rgb = torch.clamp(rgb, -1.0, 1.0)
    rgb = (rgb + 1.0) / 2.0  # → [0, 1]

    r, g, b = torch.chunk(rgb, 3, dim=1)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g - 0.10001 * b
    return torch.cat([y, u, v], dim=1)


def color_loss(con, fake):
    """
    颜色一致性损失：YUV空间
    Y通道(亮度)用L1，UV通道(色度)用Huber
    """
    con_yuv = rgb2yuv(con)
    fake_yuv = rgb2yuv(fake)
    loss_y = L1_loss(con_yuv[:, 0:1], fake_yuv[:, 0:1])
    loss_uv = Huber_loss(con_yuv[:, 1:3], fake_yuv[:, 1:3])
    return loss_y + loss_uv


def total_variation_loss(img):
    """TV正则化，减少输出图像的高频噪声"""
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).sum()
    tv_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).sum()
    return (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)


def con_sty_loss(vgg, real, anime_gray, fake):
    """
    内容 + 风格损失
    """
    content_real, _ = vgg(real)
    content_fake, style_fake = vgg(fake)
    _, style_anime = vgg(anime_gray)

    c_loss = content_loss(content_real[0], content_fake[0])
    s_loss = sum(style_loss(sa, sf) for sa, sf in zip(style_anime, style_fake))

    return c_loss, s_loss


def generator_loss(d_fake):
    """生成器对抗损失 (LSGAN)"""
    return torch.mean((d_fake - 1.0) ** 2)


def discriminator_loss(d_real, d_fake, d_gray, d_blur):
    """
    判别器对抗损失 (LSGAN)
    ✅ 核心修复 2：采用官方更平衡的权重系数。
    降低 D 的惩罚锐度，配合谱归一化，彻底解决 D 碾压 G 的问题。
    """
    loss_real = torch.mean((d_real - 1.0) ** 2)
    loss_fake = torch.mean(d_fake ** 2)
    loss_gray = torch.mean(d_gray ** 2)
    loss_blur = torch.mean(d_blur ** 2)

    # 将原来的 1.7 降调至标准的 1.2
    return 1.2 * loss_real + 1.2 * loss_fake + 1.2 * loss_gray + 0.5 * loss_blur