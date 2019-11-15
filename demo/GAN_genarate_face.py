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
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5"

np.random.seed(0)  # random seed
BATCH_SIZE=9
latent_size=64

if __name__ == '__main__':
	file_dir = "/home1/yixu/yixu_project/CVAE-GAN/download_script/download"

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	cuda = True if torch.cuda.is_available() else False
	Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

	data = read_img.get_file(file_dir)
	data = data.to(device)
	# dataset = torch.from_numpy(data)
	dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
	# for batch_idx, data in enumerate(dataloader):
	# 	real_images = data
	# 	batch_size = real_images.size(0)
	# print('#{} has {} images'.format(batch_idx, batch_size))
	# if batch_idx == 1:
	# 	path = '/home1/yixu/yixu_project/CVAE-GAN/output/save{:03d}.jpg'.format(batch_idx)
	# 	save_image(real_images, path, normalize=True)

	# print(dataloader)
	n_channel = 3
	n_g_feature = 128

	# ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0,groups=1, bias=True, dilation=1) **
	# output=(input-1)*stride+output_padding -2*padding+kernel_size

	G_net = nn.Sequential(nn.ConvTranspose2d(latent_size, 16 * n_g_feature, kernel_size=4, bias=False),
						  nn.BatchNorm2d(16 * n_g_feature),
						  nn.ReLU(),
						  nn.ConvTranspose2d(16 * n_g_feature, 8 * n_g_feature, kernel_size=4, padding=1, stride=2,
											 bias=False),
						  nn.BatchNorm2d(8 * n_g_feature),
						  nn.ReLU(),
						  nn.ConvTranspose2d(8 * n_g_feature, 4*n_g_feature, kernel_size=4, stride=2, padding=1,
											 bias=False),
						  nn.BatchNorm2d(4*n_g_feature),
						  nn.ReLU(),
						  nn.ConvTranspose2d(4*n_g_feature, 2*n_g_feature, kernel_size=4, stride=2, padding=1,bias=False),
						  nn.BatchNorm2d(2*n_g_feature),
						  nn.ReLU(),
						  nn.ConvTranspose2d(2*n_g_feature, n_g_feature, kernel_size=4, stride=2, padding=1,bias=False),
						  nn.BatchNorm2d(n_g_feature),
						  nn.ReLU(),
						  nn.ConvTranspose2d(n_g_feature, n_channel, kernel_size=4, stride=2, padding=1,bias=False),
						  nn.Sigmoid(),
						  )

	print(G_net)

	n_d_feature = 128

	D_net = nn.Sequential(nn.Conv2d(n_channel, 16*n_d_feature, kernel_size=4, stride=2, padding=1, bias=False),
						  nn.BatchNorm2d(16*n_d_feature),
						  nn.ReLU(),
						  nn.Conv2d(16*n_d_feature, 8 * n_d_feature, kernel_size=4, stride=2, padding=1, bias=False),
						  nn.BatchNorm2d(8 * n_d_feature),
						  nn.ReLU(),
						  nn.Conv2d(8 * n_d_feature, 4 * n_d_feature, kernel_size=4, padding=1, stride=2, bias=False),
						  nn.BatchNorm2d(4 * n_d_feature),
						  nn.ReLU(),
						  nn.Conv2d(4 * n_d_feature, 2 * n_d_feature, kernel_size=4, padding=1, stride=2, bias=False),
						  nn.BatchNorm2d(2 * n_d_feature),
						  nn.ReLU(),
						  nn.Conv2d(2 * n_d_feature, 1 * n_d_feature, kernel_size=4, padding=1, stride=2, bias=False),
						  nn.BatchNorm2d(1 * n_d_feature),
						  nn.ReLU(),
						  nn.Conv2d(1 * n_d_feature, 1, kernel_size=4),
						  )

	criterion = nn.BCEWithLogitsLoss() # ???/how to deal with it

	if cuda:
		G_net.cuda()
		D_net.cuda()
		criterion.cuda()

	def weights_init(m):
		if type(m) in [nn.ConvTranspose2d, nn.Conv2d]:
			nn.init.xavier_normal_(m.weight)
		elif type(m) == nn.BatchNorm2d:
			nn.init.normal(m.weight, 1.0, 0.02)
			nn.init.constant_(m.bias, 0)


	print(D_net)

	G_net.apply(weights_init)
	D_net.apply(weights_init)

	G_optimizer = torch.optim.Adam(G_net.parameters(), lr=0.0002, betas=(0.5, 0.999))  # ???
	D_optimizer = torch.optim.Adam(D_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
	fix_noises = torch.randn(BATCH_SIZE, latent_size, 1, 1).cuda()

	epoch_num = 4000
	for epoch in range(epoch_num):
		for batch_idx, data in enumerate(dataloader):
			# get data
			real_img = data
			batch_size = real_img.size(0)
			# train D
			labels_real = torch.ones(batch_size).cuda()
			preds = D_net(real_img)
			outputs = preds.reshape(-1)  # ???
			d_loss_real = criterion(outputs, labels_real)
			d_mean_real = outputs.sigmoid().mean()

			noises = torch.randn(batch_size, latent_size, 1, 1).cuda()

			fake_image = G_net(noises).cuda()
			labels_fake = torch.zeros(batch_size).cuda()

			# no_use = fake_image.detach()
			preds_fake = D_net(fake_image).cuda()

			outputs_fake = preds_fake.view(-1)#???
			d_loss_fake = criterion(outputs_fake, labels_fake)
			d_mean_fake = outputs_fake.sigmoid().mean()

			d_loss = d_loss_fake + d_loss_real
			D_net.zero_grad()
			d_loss.backward(retain_graph=True) # it will have a line, but it works
			D_optimizer.step()

			labels_gen = torch.ones(batch_size).cuda()

			preds_gen = D_net(fake_image).cuda()
			outputs_gen = preds_gen.view(-1)
			g_loss = criterion(outputs_gen, labels_gen)# problem

			g_mean_fake = outputs_gen.sigmoid().mean()

			G_net.zero_grad()
			g_loss.backward()
			G_optimizer.step()

			print('[{}/{}]'.format(epoch,epoch_num) +
				  '[{}/{}]'.format(batch_idx,len(dataloader)) +
				  'D_loss:{:g} g_loss:{:g} '.format(d_loss,g_loss) +
				  'real_img find:{:g} fake_img find,fake for real: {:g} ,real for fake:{:g}'.format(d_mean_real, d_mean_fake, g_mean_fake))
			if batch_idx == 1:
				fake_img = G_net(fix_noises).cuda()
				path = '/home1/yixu/yixu_project/CVAE-GAN/output/images_epoch{:02d}_batch{:03d}.jpg'.format(epoch,batch_idx)
				save_image(fake_img, path, nrow=3,normalize=True)
