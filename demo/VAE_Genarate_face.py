# -*- encoding: utf-8 -*-
'''
@Author  :   {Yixu}
@Contact :   {xiyu111@mail.ustc.edu.cn}
@Software:   PyCharm
@File    :   VAE_Genarate_face.py
@Time    :   6/11/2019 7:05 PM
'''
'''
use gan is not good,this time use vae,may use fullconnect to de
'''

from read_data import read_img
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn as nn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class VAE_net(nn.Module):
	def __init__(self):
		super(VAE_net, self).__init__()  # jiche fathers Attribute

		self.decoder = nn.Sequential(nn.ConvTranspose2d(latent_size, 4 * n_g_feature, kernel_size=4, bias=False),
									 nn.BatchNorm2d(4 * n_g_feature),  # input 64*1*1
									 nn.LeakyReLU(),
									 nn.ConvTranspose2d(4 * n_g_feature, 2 * n_g_feature, kernel_size=4, padding=1,
														stride=2,
														bias=False),
									 nn.BatchNorm2d(2 * n_g_feature),
									 nn.LeakyReLU(),
									 nn.ConvTranspose2d(2 * n_g_feature, n_g_feature, kernel_size=4, padding=1,
														stride=2,
														bias=False),
									 nn.BatchNorm2d(n_g_feature),
									 nn.LeakyReLU(),
									 nn.ConvTranspose2d(n_g_feature, n_g_feature // 2, kernel_size=4, stride=2,
														padding=1),
									 nn.BatchNorm2d(n_g_feature // 2),
									 nn.LeakyReLU(),
									 nn.ConvTranspose2d(n_g_feature // 2, n_g_feature // 4, kernel_size=4, stride=2,
														padding=1),
									 nn.BatchNorm2d(n_g_feature // 4),
									 nn.LeakyReLU(),
									 nn.ConvTranspose2d(n_g_feature // 4, n_channel, kernel_size=4, stride=2,
														padding=1),
									 nn.Sigmoid(),  # output 3*128*128
									 ).cuda()

		self.Encoder_cal_u = nn.Linear(64 * 1 * 1, LATENT_CODE_NUM).cuda()
		self.Encoder_cal_o = nn.Linear(64 * 1 * 1, LATENT_CODE_NUM).cuda()
		self.Encoder_cal_add_u_o = nn.Linear(LATENT_CODE_NUM, 64 * 1 * 1).cuda()

		self.Encoder_net = nn.Sequential(
			nn.Conv2d(n_channel, n_d_feature, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(n_d_feature),  # input 128*128*3
			nn.ReLU(),
			nn.Conv2d(n_d_feature, 2 * n_d_feature, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(2 * n_d_feature),
			nn.ReLU(),
			nn.Conv2d(2 * n_d_feature, 4 * n_d_feature, kernel_size=4, stride=2, padding=1,
					  bias=False),
			nn.BatchNorm2d(4 * n_d_feature),
			nn.ReLU(),
			nn.Conv2d(4 * n_d_feature, 2 * n_d_feature, kernel_size=4, stride=2, padding=1,
					  bias=False),
			nn.BatchNorm2d(2 * n_d_feature),
			nn.ReLU(),
			nn.Conv2d(2 * n_d_feature, 1 * n_d_feature, kernel_size=4, stride=2, padding=1,
					  bias=False),
			nn.BatchNorm2d(1 * n_d_feature),
			nn.ReLU(),
			nn.Conv2d(1 * n_d_feature, LATENT_CODE_NUM, kernel_size=4),  # output 64 * 1 * 1
			).cuda()

		def reparameterize(self, mu, logvar):
			eps = torch.randn(mu.size(0), mu.size(1)).cuda()  #
			z = mu + eps * torch.exp(logvar / 2)
			return z.cuda()

		def forward(self, x) :
			pred1, pred2 = self.Encoder_net(img), self.Encoder_net(img)
			mu = self.Encoder_cal_u(pred1.view(pred1.size(0), -1))  # get
			logvar = self.Encoder_cal_o(pred2.view(pred2.size(0), -1))  # get
			z = self.reparameterize(mu, logvar)
			add_u_o = self.Encoder_cal_add_u_o.view(z.size(0), 64, 1, 1)
			output = self.Decoder_net(add_u_o)  # get
			return mu.cuda(),logvar.cuda(),output.cuda()

def loss_func(recon_x,x,mu,logvar):
	BCE = F.binary_cross_entropy(recon_x,x,size_average=False)
	KLD = -0.5 * torch.sum(1+logvar - mu.pow(2) - logvar.exp())
	return BCE + KLD

vae = VAE_net().cuda()
optimizer = optim

if __name__ == '__main__':
	file_dir = "/home1/yixu/yixu_project/CVAE-GAN/download_script/download"
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	cuda = True if torch.cuda.is_available() else False
	Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor  # ????
	data = read_img.get_file(file_dir)
	data = data.to(device)

	dataloader = DataLoader(data, batch_size=64, shuffle=True)
	latent_size = 64
	n_channel = 3
	n_g_feature = 128
	LATENT_CODE_NUM = 64  # for VAE latne

	# for this time decoder is genarate
	# (self, in_channels, out_channels, kernel_size, stride=1,padding=0, output_padding=0, groups=1, bias=True,dilation=1, padding_mode='zeros'):
	# output=(input-1)*stride+output_padding -2*padding+kernel_size


	n_d_feature = 128
	# not same as up
	# output=(input-kernel_size+2*Padding)/stride + 1







	if cuda:
		Decoder_net.cuda()
		Encoder_net.cuda()


	def weights_init(m):
		if type(m) in [nn.ConvTranspose2d, nn.Conv2d]:
			nn.init.xavier_normal_(m.weight)
		elif type(m) == nn.BatchNorm2d:
			nn.init.normal(m.weight, 1.0, 0.02)
			nn.init.constant_(m.bias, 0)
		elif type(m) == nn.Linear:
			nn.init.normal(m.weight, 1.0, 0.02)
			nn.init.constant_(m.bias, 0)


	print(Decoder_net)

	Decoder_net.apply(weights_init)
	Encoder_net.apply(weights_init)
	Encoder_cal_add_u_o.apply(weights_init)  # ???
	Encoder_cal_o.apply(weights_init)
	Encoder_cal_u.apply(weights_init)





	epoch_num = 4000
	for epoch in range(epoch_num):
		for batch_idx, data in enumerate(dataloader):
			# get data
			img = data
			batch_size = img.size(0)
			total_loss = 0
			optimizer.zero_grad()

			loss = loss(output, img, mu, logvar)
