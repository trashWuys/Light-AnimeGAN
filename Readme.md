# Light-AnimeGAN
## Introduction
This project is my homework for Deeplearning lesson. Refering to the AnimeGANv2 project (https://tachibanayoshino.github.io/AnimeGANv2/), we rewrite it to Pytorch framework and do some differences.

Regarding the repid convergence of Discriminator, we optimized loss function and added spectral normalization to avoid it becoming too strong. Besides, we introduced a dynamic learning rate decay to make the generated figure more smooth and natrual.

## Dataset
The training dataset we used is from Xin Chen and Gang Liu's work (https://github.com/TachibanaYoshino/AnimeGAN/releases/tag/dataset-1). Here we focused on the Shinkai style and only trained it on this datasets. In addition, if you want to try it on other anime styles, we left a program (cut_video.py) to craete training dataset from anime videos.
Additionally, the pre=trained vgg19 weight file can be downloaded at https://download.pytorch.org/models/vgg19-dcbb9e9d.pth.

## Contact
Here we shows some contact results. 

<img src=".\Contact\1.png">  
<img src=".\Contact\2.png">

## Declaration:
This project is for learning purposes only and does not involve any commercial use.

Author:
trashGuys
