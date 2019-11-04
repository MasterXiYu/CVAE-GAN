# -*- encoding: utf-8 -*-
'''
@Author  :   {Yixu}
@Contact :   {xiyu111@mail.ustc.edu.cn}
@Software:   PyCharm
@File    :   GAN_genarate_face.py
@Time    :   4/11/2019 8:27 PM
'''

from read_data import read_img
import torch.nn as nn

if __name__ == '__main__':
    file_dir="/home1/yixu/yixu_project/CVAE-GAN/download_script/download"
    data=read_img.get_file(file_dir)




