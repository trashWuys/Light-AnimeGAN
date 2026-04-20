import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm

from net.generator import G_net
from net.discriminator import D_net
from tools.vgg19 import VGG19
from tools.ops import (
    content_loss,
    color_loss,
    total_variation_loss,
    con_sty_loss,
    generator_loss,
    discriminator_loss,
)

# ---------------------------------------------------------
# 工具函数
# ---------------------------------------------------------

def rgb_to_grayscale(img):
    """
    输入:  (B, 3, H, W), 范围 [-1, 1]
    输出:  (B, 3, H, W), 三通道相同灰度图, 范围 [-1, 1]
    """
    img = img.clamp(-1.0, 1.0)
    img_01 = (img + 1.0) / 2.0
    r, g, b = img_01[:, 0:1], img_01[:, 1:2], img_01[:, 2:3]
    gray_01 = 0.299 * r + 0.587 * g + 0.114 * b
    gray = gray_01.repeat(1, 3, 1, 1)
    gray = gray * 2.0 - 1.0
    return gray

def set_requires_grad(net, flag=True):
    for p in net.parameters():
        p.requires_grad = flag

def save_checkpoint(path, epoch, G, D, g_optimizer, d_optimizer):
    torch.save({
        "epoch": epoch,
        "G": G.state_dict(),
        "D": D.state_dict(),
        "g_optimizer": g_optimizer.state_dict(),
        "d_optimizer": d_optimizer.state_dict(),
    }, path)

def load_checkpoint(path, device, G, D, g_optimizer, d_optimizer):
    ckpt = torch.load(path, map_location=device)
    G.load_state_dict(ckpt["G"])
    D.load_state_dict(ckpt["D"])
    g_optimizer.load_state_dict(ckpt["g_optimizer"])
    d_optimizer.load_state_dict(ckpt["d_optimizer"])
    return ckpt["epoch"]

# ---------------------------------------------------------
# AnimeGANv2 主类
# ---------------------------------------------------------

class AnimeGANv2:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.checkpoint_dir = args.checkpoint_dir
        self.sample_dir = args.sample_dir
        self.epochs = args.epoch
        self.init_epochs = args.init_epoch

        # 损失权重
        self.g_adv_weight = args.g_adv_weight
        self.d_adv_weight = args.d_adv_weight
        self.con_weight = args.con_weight
        self.sty_weight = args.sty_weight
        self.color_weight = args.color_weight
        self.tv_weight = args.tv_weight

        # 模型
        self.G = G_net().to(self.device)
        self.D = D_net().to(self.device)
        self.vgg = VGG19().to(self.device)
        self.vgg.eval()
        set_requires_grad(self.vgg, False)

        # 优化器
        self.g_optimizer = optim.Adam(
            self.G.parameters(), lr=args.g_lr, betas=(0.5, 0.999)
        )
        self.d_optimizer = optim.Adam(
            self.D.parameters(), lr=args.d_lr, betas=(0.5, 0.999)
        )

        self.init_lr = getattr(args, "init_lr", 2e-4)
        self.base_g_lr = args.g_lr
        self.base_d_lr = args.d_lr  # ✅ 新增这行，记录 D 的基础学习率

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)

    def _set_g_lr(self, lr):
        for pg in self.g_optimizer.param_groups:
            pg["lr"] = lr

    # ✅ 新增：用于衰减 D 的学习率
    def _set_d_lr(self, lr):
        for pg in self.d_optimizer.param_groups:
            pg["lr"] = lr

    def train(self, train_loader):
        print(f"Training on device: {self.device}")

        # 取一个固定样本用于可视化
        first_batch = next(iter(train_loader))
        fixed_real_img = first_batch[0][0:1].to(self.device)
        save_image(
            ((fixed_real_img + 1.0) / 2.0).clamp(0, 1),
            os.path.join(self.sample_dir, "regional.jpg")
        )

        start_epoch = 0
        ckpt_path = os.path.join(self.checkpoint_dir, "latest.pth")
        if os.path.exists(ckpt_path):
            start_epoch = load_checkpoint(
                ckpt_path, self.device, self.G, self.D, self.g_optimizer, self.d_optimizer
            )
            print(f"Resumed from epoch {start_epoch}")

        for epoch in range(start_epoch, self.epochs):
            is_init_phase = (epoch < self.init_epochs)

            # 恢复/切换 G 学习率
            # if is_init_phase:
            #     self._set_g_lr(self.init_lr)
            # else:
            #     self._set_g_lr(self.base_g_lr)

            # 🚀 动态学习率调度策略
            if is_init_phase:
                self._set_g_lr(self.init_lr)
            else:
                # 当训练进度过半时，开始线性衰减学习率，促使细节收敛
                half_epochs = self.epochs // 2
                if epoch >= half_epochs:
                    decay_ratio = 1.0 - (epoch - half_epochs) / (self.epochs - half_epochs)
                    current_g_lr = self.base_g_lr * decay_ratio
                    current_d_lr = self.base_d_lr * decay_ratio
                    self._set_g_lr(max(current_g_lr, 1e-6))  # 设置最低阈值防止直接死掉
                    self._set_d_lr(max(current_d_lr, 1e-6))
                else:
                    self._set_g_lr(self.base_g_lr)
                    self._set_d_lr(self.base_d_lr)

            self.G.train()
            self.D.train()
            self.vgg.eval()

            pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{self.epochs}]")

            for i, (real_img, anime_img, anime_smooth) in enumerate(pbar):
                real_img = real_img.to(self.device)
                anime_img = anime_img.to(self.device)
                anime_smooth = anime_smooth.to(self.device)

                anime_gray = rgb_to_grayscale(anime_img)

                # =========================
                # 1) init 阶段：只预训练 G
                # =========================
                if is_init_phase:
                    set_requires_grad(self.D, False)
                    set_requires_grad(self.G, True)

                    self.g_optimizer.zero_grad()

                    fake_img = self.G(real_img)
                    [c_real], _ = self.vgg(real_img)
                    [c_fake], _ = self.vgg(fake_img)

                    loss_init = self.con_weight * content_loss(c_real, c_fake)
                    loss_init.backward()
                    self.g_optimizer.step()

                    pbar.set_postfix({
                        "phase": "init",
                        "loss": f"{loss_init.item():.4f}"
                    })
                    continue

                # =========================
                # 2) 训练 D
                # =========================
                set_requires_grad(self.D, True)
                set_requires_grad(self.G, False)

                self.d_optimizer.zero_grad()

                with torch.no_grad():
                    fake_img = self.G(real_img)

                d_real = self.D(anime_img)
                d_fake = self.D(fake_img)
                d_blur = self.D(anime_smooth)
                d_gray = self.D(anime_gray)

                # 如果你要严格接近原版，可以把 gray/blur 权重先降到更低
                loss_D = self.d_adv_weight * discriminator_loss(
                    d_real, d_fake, d_gray, d_blur
                )

                loss_D.backward()
                self.d_optimizer.step()

                # =========================
                # 3) 训练 G
                # =========================
                set_requires_grad(self.D, False)
                set_requires_grad(self.G, True)

                self.g_optimizer.zero_grad()

                fake_img = self.G(real_img)
                d_fake = self.D(fake_img)

                # adversarial
                loss_g_adv = self.g_adv_weight * generator_loss(d_fake)

                # content + style
                c_loss, s_loss = con_sty_loss(
                    self.vgg, real_img, anime_gray, fake_img
                )
                loss_con = self.con_weight * c_loss
                loss_sty = self.sty_weight * s_loss

                # color
                loss_col = self.color_weight * color_loss(real_img, fake_img)

                # tv
                loss_tv = self.tv_weight * total_variation_loss(fake_img)

                loss_G = loss_g_adv + loss_con + loss_sty + loss_col + loss_tv
                loss_G.backward()
                self.g_optimizer.step()

                pbar.set_postfix({
                    "D": f"{loss_D.item():.3f}",
                    "G": f"{loss_G.item():.3f}",
                    "adv": f"{loss_g_adv.item():.3f}",
                    "con": f"{loss_con.item():.3f}",
                    "sty": f"{loss_sty.item():.3f}",
                    "col": f"{loss_col.item():.3f}",
                })

            # 保存完整 checkpoint
            save_checkpoint(
                ckpt_path, epoch + 1, self.G, self.D, self.g_optimizer, self.d_optimizer
            )

            if (epoch + 1) % 2 == 0 or epoch == self.epochs - 1:
                self.save(self.checkpoint_dir, epoch + 1)
                self.sample(fixed_real_img, epoch + 1)

    @torch.no_grad()
    def sample(self, real_img, epoch):
        self.G.eval()
        fake_img = self.G(real_img)
        fake_img = ((fake_img + 1.0) / 2.0).clamp(0, 1)
        save_path = os.path.join(self.sample_dir, f"epoch_{epoch}.jpg")
        save_image(fake_img, save_path)
        self.G.train()

    def save(self, path, epoch):
        os.makedirs(path, exist_ok=True)
        torch.save(self.G.state_dict(), os.path.join(path, f"G_{epoch}.pth"))
        torch.save(self.D.state_dict(), os.path.join(path, f"D_{epoch}.pth"))