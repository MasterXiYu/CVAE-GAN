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

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
BATCH_SIZE = 8

class VAE_GAN(torch.nn.Module):
	def __init__(self):
		super(VAE_GAN, self).__init__()

		N_FEATURE = 128
		LATENT_SIZE = 64
		N_CHANNEL = 3
		LATENT_CODE_NUM = 64

		self.Encoder_net = nn.Sequential(
			nn.Conv2d(N_CHANNEL, 8 * N_FEATURE, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(8 * N_FEATURE),  # 64*64
			nn.ReLU(),
			nn.Conv2d(8 * N_FEATURE, 4 * N_FEATURE, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(4 * N_FEATURE),  # 32*32
			nn.ReLU(),
			nn.Conv2d(4 * N_FEATURE, 2 * N_FEATURE, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(2 * N_FEATURE),  # 16*16
			nn.ReLU(),
			nn.Conv2d(2 * N_FEATURE, 1 * N_FEATURE, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(1 * N_FEATURE),  # 8*8
			nn.ReLU(),
			nn.Conv2d(N_FEATURE, N_FEATURE, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(N_FEATURE),  # 4*4
			nn.ReLU(),
			nn.Conv2d(N_FEATURE, LATENT_CODE_NUM, kernel_size=4),  # 1*1
		).cuda()
		self.Encoder_cal_u = nn.Linear(LATENT_CODE_NUM * 1 * 1, LATENT_CODE_NUM).cuda()
		self.Encoder_cal_o = nn.Linear(LATENT_CODE_NUM * 1 * 1, LATENT_CODE_NUM).cuda()
		self.Encoder_cal_add_u_o = nn.Linear(LATENT_CODE_NUM, 64 * 1 * 1).cuda()

		self.Decoder_net = nn.Sequential(nn.ConvTranspose2d(LATENT_CODE_NUM, 16 * N_FEATURE, kernel_size=4, bias=False),
										 nn.BatchNorm2d(16 * N_FEATURE),
										 nn.LeakyReLU(),
										 nn.ConvTranspose2d(16 * N_FEATURE, 8 * N_FEATURE, kernel_size=4, padding=1,
															stride=2, bias=False),
										 nn.BatchNorm2d(8 * N_FEATURE),
										 nn.LeakyReLU(),
										 nn.ConvTranspose2d(8 * N_FEATURE, 4 * N_FEATURE, kernel_size=4, padding=1,
															stride=2, bias=False),
										 nn.BatchNorm2d(4 * N_FEATURE),
										 nn.LeakyReLU(),
										 nn.ConvTranspose2d(4 * N_FEATURE, 2 * N_FEATURE, kernel_size=4, padding=1,
															stride=2, bias=False),
										 nn.BatchNorm2d(2 * N_FEATURE),
										 nn.LeakyReLU(),
										 nn.ConvTranspose2d(2 * N_FEATURE, N_FEATURE, kernel_size=4, padding=1,
															stride=2, bias=False),
										 nn.BatchNorm2d(N_FEATURE),
										 nn.LeakyReLU(),
										 nn.ConvTranspose2d(N_FEATURE, N_CHANNEL, kernel_size=4, padding=1,
															stride=2, bias=False),
										 nn.Sigmoid(),
										 ).cuda()

		self.Discriminate = nn.Sequential(
			nn.Conv2d(N_CHANNEL, 16 * N_FEATURE, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(16 * N_FEATURE),
			nn.ReLU(),
			nn.Conv2d(16 * N_FEATURE, 8 * N_FEATURE, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(8 * N_FEATURE),
			nn.ReLU(),
			nn.Conv2d(8 * N_FEATURE, 4 * N_FEATURE, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(4 * N_FEATURE),
			nn.ReLU(),
			nn.Conv2d(4 * N_FEATURE, 2 * N_FEATURE, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(2 * N_FEATURE),
			nn.ReLU(),
			nn.Conv2d(2 * N_FEATURE, N_FEATURE, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(N_FEATURE),
			nn.ReLU(),
			nn.Conv2d(N_FEATURE, 1, kernel_size=4),
		)

	def reparameterize(self, mu, logvar):
		eps = torch.randn(mu.size(0), mu.size(1)).cuda()
		z = mu + eps * torch.exp(logvar / 2)
		return z.cuda()

	def forward(self, img):
		pred1, pred2 = self.Encoder_net(img), self.Encoder_net(img)
		mu = self.Encoder_cal_u(pred1.view(pred1.size(0), -1))
		logvar = self.Encoder_cal_o(pred2.view(pred2.size(0), -1))
		z = self.reparameterize(mu, logvar)
		add_u_o = self.Encoder_cal_add_u_o(z).view(z.size(0), 64, 1, 1)
		noises = torch.randn(z.size(0), 64, 1, 1).cuda()
		constant_img = self.Decoder_net(add_u_o)
		fake_img = self.Decoder_net(noises)
		output_real = self.Discriminate(img)
		output_fake = self.Discriminate(fake_img)
		output_constant = self.Discriminate(constant_img)
		return fake_img.cuda(), mu.cuda(), logvar.cuda(), output_fake.cuda(), output_real.cuda(), constant_img.cuda(), output_constant.cuda()


def loss_VAE(fake_img, real_img, mu, logvar, output_fake, output_real, constant_img, output_constant):
	batch_size = real_img.size(0)
	labels_real = torch.ones(batch_size).cuda()
	labels_fake = torch.zeros(batch_size).cuda()
	labels_GAN = torch.ones(batch_size).cuda()
	BCE = F.binary_cross_entropy(constant_img, real_img, size_average=False)
	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	criterion = nn.BCEWithLogitsLoss()
	D_loss_real = criterion(output_real.view(-1), labels_real)
	D_mean_real = output_real.view(-1).sigmoid().mean()
	D_loss_fake = criterion(output_fake.view(-1), labels_fake)
	D_mean_fake = output_fake.view(-1).sigmoid().mean()

	G_loss = criterion(output_fake.view(-1), labels_GAN)
	G_mean_fake = output_fake.view(-1).sigmoid().mean()

	print('D_loss:{:g} g_loss:{:g} '.format(D_loss_real + D_loss_real, G_loss) +
		  'real_img find:{:g} fake_img find: {:g} ,fake_become_real:{:g}'.format(D_mean_real, D_mean_fake, G_mean_fake))
	if torch.rand(1)[0] <= 0.05:
		return BCE + KLD + D_loss_fake + D_loss_real + G_loss
	else:  # only 10% to updata Discriminate
		return BCE + KLD + G_loss


criterion = nn.BCEWithLogitsLoss()

vae_gan = VAE_GAN().cuda()
optimizer = torch.optim.Adam(vae_gan.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

if __name__ == '__main__':
	file_dir = "/home1/yixu/yixu_project/CVAE-GAN/download_script/download"
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	cuda = True if torch.cuda.is_available() else False
	Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor  # ????
	data = read_img.get_file(file_dir)
	data = data.to(device)
	dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)


	def weights_init(m):
		if type(m) in [nn.ConvTranspose2d, nn.Conv2d]:
			nn.init.xavier_normal_(m.weight)
		elif type(m) == nn.BatchNorm2d:
			nn.init.normal(m.weight, 1.0, 0.02)
			nn.init.constant_(m.bias, 0)
		elif type(m) == nn.Linear:
			nn.init.normal(m.weight, 1.0, 0.02)
			nn.init.constant_(m.bias, 0)


	fixed_noise = torch.randn( BATCH_SIZE, 64, 1, 1).cuda()  # fix it as one
	epoch_num = 4000

	for epoch in range(epoch_num):
		for batch_idx, data in enumerate(dataloader):
			img_real = data.cuda()
			batch_size = img_real.size(0)
			optimizer.zero_grad()
			# return fake_img.cuda(), mu.cuda(), logvar.cuda(), output_fake.cuda(), output_real.cuda(), constant_img.cuda(), output_constant.cuda()
			fake_img, mu, logvar, output_fake, output_real, constant_img, output_constant = vae_gan.forward(img_real)
			loss = loss_VAE(fake_img=fake_img, real_img=img_real, mu=mu, logvar=logvar, output_fake=output_fake,
							output_real=output_real, constant_img=constant_img, output_constant=output_constant)
			loss.backward()
			optimizer.step()

			print('[{}/{}]'.format(epoch, epoch_num) +
				  '[{}/{}]'.format(batch_idx, len(dataloader)) +
				  'loss:{:g}'.format(loss))

			if batch_idx == 1:
				fake_img = vae_gan.Decoder_net(fixed_noise).cuda()
				path = '/home1/yixu/yixu_project/CVAE-GAN/output_GAN_VAE/images_epoch{:02d}_batch{:03d}.jpg'.format(
					epoch, batch_idx)
				save_image(fake_img, path, normalize=True)
