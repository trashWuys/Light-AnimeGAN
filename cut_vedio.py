import cv2
import os
import random
import numpy as np

# ===== 参数设置 =====
video_path = r"E:\DeepLearning_Homework\JOJO_datasets\源视频\D-03奇妙5。01-39\JOJO的奇妙冒险 第五部(黄金之风)01-39\第37话 720P.mp4"
output_dir = r"E:\DeepLearning_Homework\JOJO_datasets\style"
num_samples = 7000
crop_size = 256

# 避开区域比例（可调整）
top_crop_ratio = 0.15    # 去掉上方（防右上角水印）
bottom_crop_ratio = 0.1 # 去掉底部字幕
right_crop_ratio = 0.15  # 去掉右侧（水印区域）

# ==================

def extract_random_patches():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("无法打开视频")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 跳过前3分钟
    start_frame = int(fps * 180)
    end_frame = total_frames - int(fps * 180)

    if start_frame >= total_frames:
        print("视频长度不足3分钟")
        return

    # 可用帧范围
    valid_frames = list(range(start_frame, end_frame))

    # 随机选取帧
    selected_frames = random.sample(valid_frames, min(num_samples, len(valid_frames)))

    count = 6500

    for frame_id in selected_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()

        if not ret:
            continue

        h, w, _ = frame.shape

        # ===== 去除水印和字幕区域 =====
        x1 = 0
        y1 = int(h * top_crop_ratio)
        x2 = int(w * (1 - right_crop_ratio))
        y2 = int(h * (1 - bottom_crop_ratio))

        clean_region = frame[y1:y2, x1:x2]

        ch, cw, _ = clean_region.shape

        # 防止尺寸不够裁剪
        if ch < crop_size or cw < crop_size:
            continue

        # ===== 随机裁剪256x256 =====
        rx = random.randint(0, cw - crop_size)
        ry = random.randint(0, ch - crop_size)

        patch = clean_region[ry:ry+crop_size, rx:rx+crop_size]

        # 保存
        save_path = os.path.join(output_dir, f"img_{count:04d}.jpg")
        print(save_path)
        ok = cv2.imwrite(save_path, patch)
        print(save_path, "写入成功" if ok else "写入失败")

        count += 1

        if count >= num_samples:
            break

    cap.release()
    print(f"完成，共生成 {count} 张图片")

if __name__ == "__main__":
    extract_random_patches()