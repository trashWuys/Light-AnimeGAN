import os
import glob
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

# 确保导入路径正确，根据你的目录结构，可能需要 from net.generator import G_net
from net.generator import G_net

# =========================================================
# [配置区] 请根据你的本地路径修改
# =========================================================
CONFIG = {
    "test_dir": r"E:\DeepLearning_Homework\AnimeGANv2-master\AnimeGANv2-master\dataset\val",
    "checkpoint_path": r"E:\DeepLearning_Homework\AnimeGANv2-master\AnimeGANv2-master\checkpoint\G_101.pth",
    "save_dir": r"E:\DeepLearning_Homework\AnimeGANv2-master\AnimeGANv2-master\results\test_Shinkai",
    "input_size": (256, 256),  # 必须与训练时的输入尺寸一致
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


def main():
    # 1. 检查并创建输出目录
    if not os.path.exists(CONFIG["save_dir"]):
        os.makedirs(CONFIG["save_dir"])
        print(f"[*] 创建目录: {CONFIG['save_dir']}")

    # 2. 获取图片列表 (支持多种常见格式)
    test_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        test_files.extend(glob.glob(os.path.join(CONFIG["test_dir"], ext)))
        test_files.extend(glob.glob(os.path.join(CONFIG["test_dir"], ext.upper())))

    if not test_files:
        print(f"[!] 错误：在 {CONFIG['test_dir']} 没找到图片，请检查路径。")
        return

    # 3. 初始化模型并加载权重
    device = torch.device(CONFIG["device"])
    model = G_net().to(device)

    if os.path.exists(CONFIG["checkpoint_path"]):
        # 加载权重字典
        state_dict = torch.load(CONFIG["checkpoint_path"], map_location=device)
        model.load_state_dict(state_dict)
        print(f"[*] 成功加载权重: {CONFIG['checkpoint_path']}")
    else:
        print(f"[!] 找不到权重文件: {CONFIG['checkpoint_path']}")
        return

    model.eval()  # 必须切换到推理模式

    # 4. 预处理定义
    # 按照 AnimeGANv2 的逻辑，通常将输入映射到 [-1, 1]
    transform = transforms.Compose([
        transforms.Resize(CONFIG["input_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 5. 推理循环
    print(f"[*] 开始处理，共 {len(test_files)} 张图片，设备: {device}...")

    with torch.no_grad():
        for sample_file in tqdm(test_files):
            try:
                # 读取原图
                original_img = Image.open(sample_file).convert('RGB')
                w, h = original_img.size

                # 预处理: [3, H, W] -> [1, 3, 256, 256]
                input_tensor = transform(original_img).unsqueeze(0).to(device)

                # 模型推理
                fake_tensor = model(input_tensor)

                # --- 还原至原尺度 ---
                # 使用双线性插值将生成的图像放大回原图尺寸 (h, w)
                fake_rescaled = F.interpolate(fake_tensor, size=(h, w), mode='bilinear', align_corners=False)

                # 保存图片
                save_name = os.path.basename(sample_file)
                # 如果想区分文件名，可以加上后缀：save_name = os.path.splitext(save_name)[0] + "_anime.jpg"
                save_path = os.path.join(CONFIG["save_dir"], save_name)

                # save_image 关键参数：
                # normalize=True 会自动将输入的 [-1, 1] 映射到 [0, 1] 以便保存
                save_image(fake_rescaled, save_path, normalize=True, value_range=(-1, 1))

            except Exception as e:
                print(f"[!] 处理 {sample_file} 时出错: {e}")

    print(f"[*] 处理完成！结果保存在: {CONFIG['save_dir']}")


if __name__ == '__main__':
    main()