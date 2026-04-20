import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob


class AnimeDataset(Dataset):
    def __init__(self, data_dir, dataset_name, size=[256, 256]):
        self.size = size

        # 核心修改：根据你的描述，train_photo 直接在 dataset 下
        # 而风格图（style）和平滑图（smooth）在 dataset/数据集名称 下
        self.photo_path = os.path.join(data_dir, 'train_photo')
        self.style_path = os.path.join(data_dir, dataset_name, 'style')
        self.smooth_path = os.path.join(data_dir, dataset_name, 'smooth')

        # 获取文件列表
        self.photo_files = glob(os.path.join(self.photo_path, '*.*'))
        self.style_files = glob(os.path.join(self.style_path, '*.*'))
        self.smooth_files = glob(os.path.join(self.smooth_path, '*.*'))

        if len(self.photo_files) == 0:
            print(f"警告: 未在 {self.photo_path} 找到图片，请检查路径。")

        self.len = len(self.photo_files)

    def __len__(self):
        return self.len

    def _load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size[1], self.size[0]))
        # 归一化到 [-1, 1]
        img = (img.astype(np.float32) / 127.5) - 1.0
        # 转换为 PyTorch 的 [C, H, W]
        return torch.from_numpy(img).permute(2, 0, 1)

    def __getitem__(self, index):
        # 真实照片按索引获取
        photo = self._load_image(self.photo_files[index])

        # 风格图和平滑图随机获取（解决样本数量不一致的问题）
        style_idx = np.random.randint(0, len(self.style_files))
        smooth_idx = np.random.randint(0, len(self.smooth_files))

        style = self._load_image(self.style_files[style_idx])
        smooth = self._load_image(self.smooth_files[smooth_idx])

        return photo, style, smooth