# -*- encoding: utf-8 -*-
'''
@Author  :   {Yixu}
@Contact :   {xiyu111@mail.ustc.edu.cn}
@Software:   PyCharm
@File    :   DCGAN_CelebA_pre.py
@Time    :   2/12/2019 11:17 AM
'''
import os
import matplotlib.pyplot as plt
import cv2

# root path depends on your computer
root = 'CelebA/Img/img_align_celeba_png/img_align_celeba_png.7z/img_align_celeba_png/'
save_root = 'resized_celebA/'
resize_size = 64

if not os.path.isdir(save_root):
    os.mkdir(save_root)
if not os.path.isdir(save_root + 'celebA'):
    os.mkdir(save_root + 'celebA')
img_list = os.listdir(root)

# ten_percent = len(img_list) // 10

for i in range(len(img_list)):
    img = plt.imread(root + img_list[i])
    img = img[20:198,0:178]
    img = cv2.resize(img, (resize_size, resize_size), interpolation=cv2.INTER_LINEAR)
    plt.imsave(fname=save_root + 'celebA/' + img_list[i], arr=img)

    if (i % 1000) == 0:
        print('%d images complete' % i)