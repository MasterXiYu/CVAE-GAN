# -*- encoding: utf-8 -*-
'''
@Author  :   {Yixu}
@Contact :   {xiyu111@mail.ustc.edu.cn}
@Software:   PyCharm
@File    :   read_img.py
@Time    :   4/11/2019 8:29 PM
'''
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transform

def Normalization(data):
	return data/255.0

def Change_channel(img):
	assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = Normalization(img)
	img = img.transpose((2, 0, 1))
	# img = torch.from_numpy(img.transpose((2, 0, 1)))
	img = img.astype(np.float32)
	return img

def get_file(file_dir):  # file_dir="/home1/yixu/yixu_project/CVAE-GAN/download_script/download"
	class_train = []
	lable_train = []
	for path in os.listdir(file_dir):
		face_dir = os.path.join(file_dir, path)
		face_128_dir = os.path.join(face_dir, "face_128*128")
		for file in os.listdir(face_128_dir):
			face_file = os.path.join(face_128_dir, file)
			img = cv2.imread(face_file)
			img_array = np.array(img,dtype='float32')
			img_array = Change_channel(img_array)
			class_train.append(img_array)  # img as data
			lable_train.append(path)  # dir as lable
	# let us do not care label first

	# np.random.shuffle(class_train)
	temp = np.array(class_train) # trun to numpy.data
	temp_1 = torch.from_numpy(temp)
	return temp_1
