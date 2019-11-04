# -*- encoding: utf-8 -*-
'''
@Author  :   {Yixu}
@Contact :   {xiyu111@mail.ustc.edu.cn}
@Software:   PyCharm
@File    :   cut_script.py
@Time    :   4/11/2019 5:24 PM

cut face to 128*128
'''

import cv2
import os
from tqdm import tqdm

if __name__ == '__main__':
	file_dir = os.getcwd() + "/download"
	file_path = os.listdir(file_dir)
	for path in tqdm(file_path):
		m = os.path.join(file_dir, path)
		if not os.path.exists(m + "/face_128*128"):# use for debug
			os.makedirs(m + "/face_128*128")
		m_child = os.path.join(m, 'face')
		file_path_face = os.listdir(m_child)
		for i in file_path_face:
			temp_file = os.path.join(m_child, i)
			img = cv2.imread(temp_file)
			try: #I don't know why but if there have no try ,it comes an error
				img_new = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
				# print(m + "/face_128*128/" + i)
				if not os.path.exists(m + "/face_128*128/" + i):
					cv2.imwrite(m + "/face_128*128/" + i, img_new)
			except:
				print("error")
print("done")
