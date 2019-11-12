# -*- encoding: utf-8 -*-
'''
@Author  :   {Yixu}
@Contact :   {xiyu111@mail.ustc.edu.cn}
@Software:   PyCharm
@File    :   GAN_VAE_genarate.py
@Time    :   12/11/2019 10:53 AM
'''

# combine GAN and VAE
from read_data import read_img
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn as nn
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class VAE_GAN(torch.nn.Module):
	def __init__(self):
		super(VAE_GAN,self).__init__()

		N_FEATURE = 128
		LATENT_SIZE = 64
		N_CHANNEL = 3
		LATENT_CODE_NUM = 64

		self.Encoder_net = nn.Sequential(nn.Conv2d(N_CHANNEL,N_FEATURE,kernel_size=4,stride=2,padding=1,bias = False),
										 nn.BatchNorm2d(N_FEATURE),
										 nn.ReLU(),
										 nn.Conv2d(N_FEATURE,2*N_FEATURE,kernel_size=4,stride=2,padding=1,bias = False),
										 nn.BatchNorm2d(2*N_FEATURE),
										 nn.ReLU(),
										 nn.Conv2d(2*N_FEATURE,4*N_FEATURE,kernel_size=4,stride=2,padding=1,bias=False),
										 nn.BatchNorm2d(4*N_FEATURE),
										 nn.ReLU(),
										 nn.Conv2d(4*N_FEATURE,2*N_FEATURE,kernel_size=4,stride=2,padding=1,bias=False),
										 nn.BatchNorm2d(2*N_FEATURE),
										 nn.ReLU(),
										 nn.Conv2d(2*N_FEATURE,N_FEATURE,kernel_size=4,stride=2,padding=1,bias=False),
										 nn.BatchNorm2d(N_FEATURE),
										 nn.ReLU(),
										 nn.Conv2d(N_FEATURE,N_CHANNEL,kernel_size=4,stride=2,padding=1,bias=False),
										 ).cuda()
		self.Encoder_cal_u = nn.Linear(LATENT_CODE_NUM*1*1,LATENT_CODE_NUM).cuda()
		self.Encoder_cal_o = nn.Linear(LATENT_CODE_NUM*1*1,LATENT_CODE_NUM).cuda()
		self.Encoder_cal_add_u_o = nn.Linear(LATENT_CODE_NUM,64*1*1).cuda()

		self.Decoder_net = nn.Sequential(nn.ConvTranspose2d(LATENT_CODE_NUM,4 * N_FEATURE,kernel_size=4,bias=False),
										 nn.BatchNorm2d(4 * N_FEATURE),
										 nn.LeakyReLU(),
										 nn.ConvTranspose2d(4*N_FEATURE,2*N_FEATURE,kernel_size=4,padding=1,stride=2,bias=False),
										 nn.BatchNorm2d(2*N_FEATURE),
										 nn.LeakyReLU(),
										 nn.ConvTranspose2d(2*N_FEATURE,N_FEATURE,kernel_size=4,padding=1,stride=2,bias=False),
										 nn.BatchNorm2d(N_FEATURE),
										 nn.LeakyReLU(),
										 nn.ConvTranspose2d(N_FEATURE,N_FEATURE//2,kernel_size=4,padding=2,stride=2,bias=False),
										 nn.BatchNorm2d(N_FEATURE//2),
										 nn.LeakyReLU(),
										 nn.ConvTranspose2d(N_FEATURE//2,N_FEATURE//4,kernel_size=4,1,stride=2,bias=False),
										 nn.BatchNorm2d(N_FEATURE//4),
										 nn.LeakyReLU(),
										 nn.ConvTranspose2d(N_FEATURE//4,N_CHANNEL,kernel_size=4,padding=1,stride=2,bias=False),
										 nn.Sigmoid(),
										 ).cuda()

		self.Discriminate = nn.Sequential(nn.Conv2d(N_CHANNEL,N_FEATURE,kernel_size=4,stride=2,padding=1,bias=False),
										  nn.BatchNorm2d(N_FEATURE),
										  nn.ReLU(),
										  nn.Conv2d(N_FEATURE,2*N_FEATURE,kernel_size=4,stride=2,padding=1,bias=False),
										  nn.BatchNorm2d(2*N_FEATURE),
										  nn.ReLU(),
										  nn.Conv2d(2*N_FEATURE,4*N_FEATURE,kernel_size=4,stride=2,padding=1,bias=False),
										  nn.BatchNorm2d(4*N_FEATURE),
										  nn.ReLU(),
										  nn.Conv2d(4*N_FEATURE,2*N_FEATURE,kernel_size=4,stride=2,padding=1,bias=False),
										  nn.BatchNorm2d(2*N_FEATURE),
										  nn.ReLU(),
										  nn.Conv2d(2*N_FEATURE,N_FEATURE,kernel_size=4,stride=2,padding=1,bias=False),
										  nn.BatchNorm2d(N_FEATURE),
										  nn.ReLU(),
										  nn.Conv2d(N_FEATURE,1,kernel_size=4),
										  )

	def reparameterize(self,mu,logvar):
		eps = torch.randn(mu.size(0),mu.size(1)).cuda()
		z=mu+eps*torch.exp(logvar/2)
		return z.cuda()

	def forward(self,img):
		pred1,pred2=self.Encoder_net(img),self.Encoder_net(img)
		mu = self.Encoder_cal_u(pred1.view(pred1.size(0),-1))
		logvar = self.Encoder_cal_o(pred2.view(pred2.size(0),-1))
		z = self.reparameterize(mu,logvar)
		add_u_o = self.Encoder_cal_add_u_o(z).view(z.size(0),64,1,1)
		fake_img = self.Decoder_net(add_u_o)
		output_real =self.Discriminate(img)
		output_fake = self.Discriminate(fake_img)
		return fake_img.cuda(),mu.cuda(),logvar.cuda(),output_fake.cuda(),output_real.cuda()

def loss_fuc(fake_img,img,mu,logvar)
