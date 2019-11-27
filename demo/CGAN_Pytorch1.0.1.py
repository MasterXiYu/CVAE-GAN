# -*- encoding: utf-8 -*-
'''
@Author  :   {Yixu}
@Contact :   {xiyu111@mail.ustc.edu.cn}
@Software:   PyCharm
@File    :   CGAn.py
@Time    :   19/11/2019 6:18 PM
'''
import torch
import torch.nn as nn

class ganeratate(nn.Module):
	def __init__(self):
		super(ganeratate,self).__init__()
		self.Z_Pre_conv = nn.Sequential(nn.Linear(in_features=100,out_features=256),
										nn.BatchNorm1d(256),
										nn.ReLU(),)
		self.Y_Pre_conv = nn.Sequential(nn.Linear(in_features=10,out_features=256),
										nn.BatchNorm1d(256),
										nn.ReLU())
		self.Z_Y_net = nn.Sequential(nn.Linear(in_features=512,out_features=512),
									 nn.BatchNorm1d(512),
									 nn.ReLU(),
									 nn.Linear(in_features=512,out_features=1024),
									 nn.BatchNorm1d(1024),
									 nn.ReLU(),
									 nn.Linear(in_features=1024,out_features=784),
									 nn.Tanh())

	def forward(self,number_image,label):
		z=self.Z_Pre_conv(number_image)
		y=self.Y_Pre_conv(label)
		z=torch.cat([z,y],1)
		reconsitution_img= self.Z_Y_net(z)
		return reconsitution_img

class discriminator(nn.Module):
	def __init__(self):
		super(discriminator,self).__init__()
		self.X_Pre_conv = nn.Sequential(nn.Linear(in_features=784,out_features=1024),
										nn.BatchNorm1d(1024),
										nn.ReLU(),
										)
		self.Y_Pre_conv = nn.Sequential(nn.Linear(in_features=10, out_features=1024),
										nn.BatchNorm1d(1024),
										nn.ReLU())
		self.X_Y_net = nn.Sequential(nn.Linear(in_features=2048, out_features=512),
									 nn.BatchNorm1d(512),
									 nn.ReLU(),
									 nn.Linear(in_features=512, out_features=1024),
									 nn.BatchNorm1d(1024),
									 nn.ReLU(),
									 nn.Linear(in_features=1024, out_features=784),
									 nn.Tanh())

