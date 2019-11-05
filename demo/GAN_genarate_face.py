# -*- encoding: utf-8 -*-
'''
@Author  :   {Yixu}
@Contact :   {xiyu111@mail.ustc.edu.cn}
@Software:   PyCharm
@File    :   GAN_genarate_face.py
@Time    :   4/11/2019 8:27 PM
'''

from read_data import read_img
from torch.utils.data import DataLoader
import torch
import numpy as np
from torchvision.utils import save_image

np.random.seed(0)  # random seed

if __name__ == '__main__':
	file_dir = "/home1/yixu/yixu_project/CVAE-GAN/download_script/download"
	data = read_img.get_file(file_dir)
	dataset = torch.from_numpy(data)
	dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
	for batch_idx, data in enumerate(dataloader):
		real_images = data
		batch_size = real_images.size(0)
		print('#{} has {} images'.format(batch_idx, batch_size))
		if batch_idx == 90:
			path = '/home1/yixu/yixu_project/CVAE-GAN/output/save{:03d}.jpg'.format(batch_idx)
			save_image(real_images, path, normalize=True)

	print(dataloader)
