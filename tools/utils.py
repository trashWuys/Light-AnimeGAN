import os
import cv2
import numpy as np
import torch
from tools.adjust_brightness import adjust_brightness_from_src_to_dst, read_img


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def load_test_data(image_path, size):
    img = cv2.imread(image_path).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocessing(img, size)  # 返回的是 [H, W, C] 且归一化后的数据
    # 转为 PyTorch 的 [1, C, H, W]
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img


def preprocessing(img, size):
    h, w = img.shape[:2]
    # 确保尺寸是 32 的倍数（对 GAN 的下采样友好）
    new_h = size[0] if h <= size[0] else h - (h % 32)
    new_w = size[1] if w < size[1] else w - (w % 32)

    img = cv2.resize(img, (new_w, new_h))
    return img / 127.5 - 1.0  # 归一化到 [-1, 1]


def save_images(tensor, image_path, photo_path=None):
    """
    tensor: [1, 3, H, W] 的 PyTorch 张量
    """
    # 1. 转回 CPU 和 Numpy
    img = tensor.detach().cpu().numpy().squeeze()
    # 2. 维度转回 [H, W, 3]
    img = img.transpose(1, 2, 0)
    # 3. 反归一化
    fake = inverse_transform(img)

    if photo_path:
        # 亮度校准（使用原项目的逻辑）
        return imsave(adjust_brightness_from_src_to_dst(fake, read_img(photo_path)), image_path)
    else:
        return imsave(fake, image_path)


def inverse_transform(images):
    # [-1, 1] -> [0, 255]
    images = (images + 1.) / 2 * 255.0
    images = np.clip(images, 0, 255)
    return images.astype(np.uint8)


def imsave(images, path):
    # 注意：OpenCV 使用 BGR 格式
    return cv2.imwrite(path, cv2.cvtColor(images, cv2.COLOR_RGB2BGR))


def random_crop(img1, img2, crop_H, crop_W):
    """
    用于训练时的数据增强
    """
    h, w = img1.shape[:2]
    x0 = np.random.randint(0, max(1, w - crop_W))
    y0 = np.random.randint(0, max(1, h - crop_H))

    crop_1 = img1[y0:y0 + crop_H, x0:x0 + crop_W]
    crop_2 = img2[y0:y0 + crop_H, x0:x0 + crop_W]
    return crop_1, crop_2