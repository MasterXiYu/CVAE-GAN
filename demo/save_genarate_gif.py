# -*- encoding: utf-8 -*-
'''
@Author  :   {Yixu}
@Contact :   {xiyu111@mail.ustc.edu.cn}
@Software:   PyCharm
@File    :   save_genarate_gif.py
@Time    :   7/11/2019 9:38 AM
'''

import imageio
import os
import re

def sort_key(name):
	c=re.search("(?<=epoch)\d+",name)
	return int(c[0])


def create_gif(image_list,gif_name):
	frames = []
	for image_name in image_list:
		if image_name.endswith('.jpg'):
			frames.append(imageio.imread(image_name))
	imageio.mimsave(gif_name,frames,'GIF')

def main():
	path = os.path.dirname(os.getcwd())
	path = os.path.join(path, 'output')
	image_list = [path+'/'+img for img in os.listdir(path)]
	image_list.sort(key = sort_key)
	gif_name = 'create_gif.gif'
	create_gif(image_list,gif_name)

if __name__ == '__main__':
	main()

