# -*- encoding: utf-8 -*-
'''
@Author  :   {Yixu}
@Contact :   {xiyu111@mail.ustc.edu.cn}
@Software:   PyCharm
@File    :   read_img.py
@Time    :   4/11/2019 8:29 PM
'''
import os
import numpy as np


def get_file(file_dir):  # file_dir="/home1/yixu/yixu_project/CVAE-GAN/download_script/download"
	class_train = []
	lable_train = []
	for path in os.listdir(file_dir):
		face_dir = os.path.join(file_dir, path)
		face_128_dir = os.path.join(face_dir, "face_128*128")
		for file in os.listdir(face_128_dir):
			face_file = os.path.join(face_128_dir, file)
			class_train.append(face_file)  # img as data
			lable_train.append(path)  # dir as lable
	# let us do not care label first
	temp = np.array([class_train])
	np.random.shuffle(temp)
	image_list = list(temp)
	return image_list
