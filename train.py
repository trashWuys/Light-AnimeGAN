from AnimeGANv2 import AnimeGANv2
import argparse
from tools.utils import *
from tools.data_loader import AnimeDataset # 确保导入了 Dataset
from torch.utils.data import DataLoader
import os
import gc
import torch # 替换 tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""parsing and configuration"""

def parse_args():
    desc = "AnimeGANv2"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', type=str, default='Shinkai', help='dataset_name')
    # parser.add_argument('--sample_dir', type=str, default='samples', help='sample_dir')

    parser.add_argument('--epoch', type=int, default=101, help='The number of epochs to run')
    parser.add_argument('--init_epoch', type=int, default=10, help='The number of epochs for weight initialization')
    parser.add_argument('--batch_size', type=int, default=12, help='The size of batch size') # if light : batch_size = 20
    parser.add_argument('--save_freq', type=int, default=1, help='The number of ckpt_save_freq')

    parser.add_argument('--init_lr', type=float, default=2e-4, help='The learning rate')
    parser.add_argument('--g_lr', type=float, default=2e-5, help='The learning rate')
    parser.add_argument('--d_lr', type=float, default=4e-5, help='The learning rate')
    parser.add_argument('--ld', type=float, default=10.0, help='The gradient penalty lambda')

    parser.add_argument('--g_adv_weight', type=float, default=200.0, help='Weight about GAN')
    parser.add_argument('--d_adv_weight', type=float, default=200.0, help='Weight about GAN')
    parser.add_argument('--con_weight', type=float, default=15, help='Weight about VGG19') # 1.5 for Hayao, 2.0 for Paprika 1.2 for Shinkai
    parser.add_argument('--sty_weight', type=float, default=3000, help='Weight about style loss') # 2.5 for Hayao 0.6 for Paprika 2.0 for Shinkai
    parser.add_argument('--color_weight', type=float, default=15.0, help='Weight about color loss') # 15 for Hayao 50 for Paprika 10 for Shinkai
    parser.add_argument('--tv_weight', type=float, default=1.0, help='Weight about tv loss') # 1 for Hayao 0.1 for Paprika 1 for Shinkai
    parser.add_argument('--training_rate', type=int, default=1, help='training rate about G & D')

    parser.add_argument('--gan_type', type=str, default='lsgan', help='[gan / lsgan / wgan-gp / wgan-lp / dragan / hinge]')

    parser.add_argument('--img_size', type=list, default=[256, 256], help='The size of image: H, W')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    parser.add_argument('--data_dir', type=str, default='dataset', help='path to dataset root')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --log_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

# 清理显存
def clear_gpu_memory():
    try:
        from numba import cuda
        cuda.select_device(0)
        cuda.close()
    except:
        pass
    gc.collect()

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # 1. 准备数据加载器
    # 传入 data_dir ('dataset') 和 dataset_name ('Hayao')
    train_dataset = AnimeDataset(args.data_dir, args.dataset, args.img_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 2. 初始化模型
    gan = AnimeGANv2(args)

    # 3. 运行训练
    gan.train(train_loader)

if __name__ == '__main__':
    print("Starting Training Process...")
    print("Clear GPU Memory Before Training...")
    clear_gpu_memory()
    main()
    print("Process Over, Clear GPU Memory...")
    clear_gpu_memory()