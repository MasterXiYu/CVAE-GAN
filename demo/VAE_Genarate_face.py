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


class VAE_net(torch.nn.Module):
	def __init__(self):
		super(VAE_net, self).__init__()  # jiche fathers Attribute
		latent_size = 64
		n_channel = 3
		n_feature = 128
		LATENT_CODE_NUM = 64  # for VAE latne

		self.Decoder_net = nn.Sequential(nn.ConvTranspose2d(latent_size, 4 * n_feature, kernel_size=4, bias=False),
										 nn.BatchNorm2d(4 * n_feature),  # input 64*1*1
										 nn.LeakyReLU(),
										 nn.ConvTranspose2d(4 * n_feature, 2 * n_feature, kernel_size=4, padding=1,
															stride=2,
															bias=False),
										 nn.BatchNorm2d(2 * n_feature),
										 nn.LeakyReLU(),
										 nn.ConvTranspose2d(2 * n_feature, n_feature, kernel_size=4, padding=1,
															stride=2,
															bias=False),
										 nn.BatchNorm2d(n_feature),
										 nn.LeakyReLU(),
										 nn.ConvTranspose2d(n_feature, n_feature // 2, kernel_size=4, stride=2,
															padding=1),
										 nn.BatchNorm2d(n_feature // 2),
										 nn.LeakyReLU(),
										 nn.ConvTranspose2d(n_feature // 2, n_feature // 4, kernel_size=4, stride=2,
															padding=1),
										 nn.BatchNorm2d(n_feature // 4),
										 nn.LeakyReLU(),
										 nn.ConvTranspose2d(n_feature // 4, n_channel, kernel_size=4, stride=2,
															padding=1),
										 nn.Sigmoid(),  # output 3*128*128
										 ).cuda()

		self.Encoder_cal_u = nn.Linear(64 * 1 * 1, LATENT_CODE_NUM).cuda()
		self.Encoder_cal_o = nn.Linear(64 * 1 * 1, LATENT_CODE_NUM).cuda()
		self.Encoder_cal_add_u_o = nn.Linear(LATENT_CODE_NUM, 64 * 1 * 1).cuda() #???

		self.Encoder_net = nn.Sequential(
			nn.Conv2d(n_channel, n_feature, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(n_feature),  # input 128*128*3
			nn.ReLU(),
			nn.Conv2d(n_feature, 2 * n_feature, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(2 * n_feature),
			nn.ReLU(),
			nn.Conv2d(2 * n_feature, 4 * n_feature, kernel_size=4, stride=2, padding=1,
					  bias=False),
			nn.BatchNorm2d(4 * n_feature),
			nn.ReLU(),
			nn.Conv2d(4 * n_feature, 2 * n_feature, kernel_size=4, stride=2, padding=1,
					  bias=False),
			nn.BatchNorm2d(2 * n_feature),
			nn.ReLU(),
			nn.Conv2d(2 * n_feature, 1 * n_feature, kernel_size=4, stride=2, padding=1,
					  bias=False),
			nn.BatchNorm2d(1 * n_feature),
			nn.ReLU(),
			nn.Conv2d(1 * n_feature, LATENT_CODE_NUM, kernel_size=4),  # output 64 * 1 * 1
		).cuda()

	def reparameterize(self, mu, logvar):
		eps = torch.randn(mu.size(0), mu.size(1)).cuda()  #
		z = mu + eps * torch.exp(logvar / 2)
		return z.cuda()

	def forward(self, img):
		pred1, pred2 = self.Encoder_net(img), self.Encoder_net(img)
		mu = self.Encoder_cal_u(pred1.view(pred1.size(0), -1))  # get
		logvar = self.Encoder_cal_o(pred2.view(pred2.size(0), -1))  # get
		z = self.reparameterize(mu, logvar)
		add_u_o = self.Encoder_cal_add_u_o(z).view(z.size(0), 64, 1, 1)
		output = self.Decoder_net(add_u_o)  # get
		return output.cuda(), mu.cuda(), logvar.cuda()


def loss_func(recon_x, x, mu, logvar):
	# BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
	# KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	# return BCE + KLD
	criterion = torch.nn.MSELoss()
	l2_loss = criterion(recon_x, x)
	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	return l2_loss + KLD


vae = VAE_net().cuda()
optimizer = torch.optim.Adam(vae.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

if __name__ == '__main__':
	file_dir = "/home1/yixu/yixu_project/CVAE-GAN/download_script/download"
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	cuda = True if torch.cuda.is_available() else False
	Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor  # ????
	data = read_img.get_file(file_dir)
	data = data.to(device)

	dataloader = DataLoader(data, batch_size=64, shuffle=True)


	# for this time decoder is genarate
	# (self, in_channels, out_channels, kernel_size, stride=1,padding=0, output_padding=0, groups=1, bias=True,dilation=1, padding_mode='zeros'):
	# output=(input-1)*stride+output_padding -2*padding+kernel_size

	# not same as up
	# output=(input-kernel_size+2*Padding)/stride + 1

	def weights_init(m):
		if type(m) in [nn.ConvTranspose2d, nn.Conv2d]:
			nn.init.xavier_normal_(m.weight)
		elif type(m) == nn.BatchNorm2d:
			nn.init.normal(m.weight, 1.0, 0.02)
			nn.init.constant_(m.bias, 0)
		elif type(m) == nn.Linear:
			nn.init.normal(m.weight, 1.0, 0.02)
			nn.init.constant_(m.bias, 0)


	#
	vae.Decoder_net.apply(weights_init)
	vae.Encoder_net.apply(weights_init)
	vae.Encoder_cal_add_u_o.apply(weights_init)  # ???
	vae.Encoder_cal_o.apply(weights_init)
	vae.Encoder_cal_u.apply(weights_init)

	fixed_noise = torch.randn(64, 64, 1, 1).cuda()  # fix it as one
	epoch_num = 4000

	for epoch in range(epoch_num):
		for batch_idx, data in enumerate(dataloader):
			# get data
			img = data.cuda()
			batch_size = img.size(0)
			total_loss = 0
			optimizer.zero_grad()
			recon_img, mu, logvar = vae.forward(img)

			loss = loss_func(recon_img, img, mu, logvar)
			loss.backward()
			optimizer.step()

			if batch_idx == 1:
				fake_img = vae.Decoder_net(fixed_noise).cuda()
				# path = '/home1/yixu/yixu_project/CVAE-GAN/output_VAE/images_epoch{:02d}_batch{:03d}.jpg'.format(epoch,batch_idx)
				path = '/home1/yixu/yixu_project/CVAE-GAN/output_VAE_l2loss/images_epoch{:02d}_batch{:03d}.jpg'.format(
					epoch, batch_idx)
				save_image(fake_img, path, normalize=True)

			print('[{}/{}]'.format(epoch, epoch_num) +
				  '[{}/{}]'.format(batch_idx, len(dataloader)) +
				  'loss:{:g}'.format(loss))
