# -*- encoding: utf-8 -*-
'''
@Author  :   {Yixu}
@Contact :   {xiyu111@mail.ustc.edu.cn}
@Software:   PyCharm
@File    :   DCGAN_CelebA.py
@Time    :   2/12/2019 11:14 AM
'''

import os, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from read_data import read_img
from torch.utils.data import DataLoader


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# G(z)
class generator(nn.Module):
	# initializers
	def __init__(self, d=128):
		super(generator, self).__init__()
		self.deconv1 = nn.ConvTranspose2d(100, d * 8, 4, 1, 0)
		self.deconv1_bn = nn.BatchNorm2d(d * 8)
		self.deconv2 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
		self.deconv2_bn = nn.BatchNorm2d(d * 4)
		self.deconv3 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
		self.deconv3_bn = nn.BatchNorm2d(d * 2)
		self.deconv4 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
		self.deconv4_bn = nn.BatchNorm2d(d)
		self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

# weight_init
	def weight_init(self, mean, std):
		for m in self._modules:
			normal_init(self._modules[m], mean, std)

	# forward method
	def forward(self, img):
		# x = F.relu(self.deconv1(input))
		x = F.relu(self.deconv1_bn(self.deconv1(img)))
		x = F.relu(self.deconv2_bn(self.deconv2(x)))
		x = F.relu(self.deconv3_bn(self.deconv3(x)))
		x = F.relu(self.deconv4_bn(self.deconv4(x)))
		x = torch.tanh(self.deconv5(x))

		return x


class discriminator(nn.Module):
	# initializers
	def __init__(self, d=128):
		super(discriminator, self).__init__()
		self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
		self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
		self.conv2_bn = nn.BatchNorm2d(d * 2)
		self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
		self.conv3_bn = nn.BatchNorm2d(d * 4)
		self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
		self.conv4_bn = nn.BatchNorm2d(d * 8)
		self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 0)

	# weight_init
	def weight_init(self, mean, std):
		for m in self._modules:
			normal_init(self._modules[m], mean, std)

	# forward method
	def forward(self, img):
		x = F.leaky_relu(self.conv1(img), 0.2)
		x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
		x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
		x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
		x = torch.sigmoid(self.conv5(x))

		return x


def normal_init(m, mean, std):
	if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
		m.weight.data.normal_(mean, std)
		m.bias.data.zero_()



def show_result(num_epoch, show=False, save=False, path = 'result.png', isFix=False):
	z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)
	z_ = z_.cuda()

	# G.eval()
	if isFix:
		test_images = G(fixed_z_)
	else:
		test_images = G(z_)
	# G.train()

	size_figure_grid = 5
	fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
	for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
		ax[i, j].get_xaxis().set_visible(False)
		ax[i, j].get_yaxis().set_visible(False)

	for k in range(5 * 5):
		i = k // 5
		j = k % 5
		ax[i, j].cla()
		ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

	label = 'Epoch {0}'.format(num_epoch)
	fig.text(0.5, 0.04, label, ha='center')
	plt.savefig(path)

	if show:
		plt.show()
	else:
		plt.close()


def show_train_hist(hist, show=False, save=False, path='Train_hist.png'):
	x = range(len(hist['D_losses']))

	y1 = hist['D_losses']
	y2 = hist['G_losses']

	plt.plot(x, y1, label='D_loss')
	plt.plot(x, y2, label='G_loss')

	plt.xlabel('Iter')
	plt.ylabel('Loss')

	plt.legend(loc=4)
	plt.grid(True)
	plt.tight_layout()

	if save:
		plt.savefig(path)

	if show:
		plt.show()
	else:
		plt.close()


# training parameters
batch_size = 128
lr = 0.0002
train_epoch = 20

# data_loader
img_size = 64
isCrop = False
if isCrop:
	transform = transforms.Compose([
		transforms.Scale(108),
		transforms.ToTensor(),
		transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
	])
else:
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
	])
# transform = transforms.Compose([
# 	transforms.ToTensor(),
# 	transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# ])

data_dir = '/home1/yixu/yixu_project/CVAE-GAN/resized_celebA/celebA'  # this path depends on your computer
data = read_img.get_file(data_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)
train_loader  = DataLoader(data, batch_size=16, shuffle=True)

# dset = datasets.ImageFolder(root=data_dir, transform=transform)
# train_loader = torch.utils.data.DataLoader(dset, batch_size=128, shuffle=True)



fixed_z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)  # fixed noise 25*100
fixed_z_ = fixed_z_.cuda()
# temp = plt.imread(train_loader.dataset.imgs[0][0])
# if (temp.shape[0] != img_size) or (temp.shape[0] != img_size):
# 	sys.stderr.write('Error! image size is not 64 x 64! run \"celebA_data_preprocess.py\" !!!')
# 	sys.exit(1)

def weights_init(m):
	if type(m) in [nn.ConvTranspose2d, nn.Conv2d]:
		nn.init.xavier_normal_(m.weight)
	elif type(m) == nn.BatchNorm2d:
		nn.init.normal(m.weight, 1.0, 0.02)
		nn.init.constant_(m.bias, 0)
	elif type(m) == nn.Linear:
		nn.init.normal(m.weight, 1.0, 0.02)
		nn.init.constant_(m.bias, 0)

G = generator()
D = discriminator()
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G.cuda()
D.cuda()
if torch.cuda.device_count()>1:
	G = nn.DataParallel(G)
	D = nn.DataParallel(D)
#
# model_dir_G = '/home1/yixu/yixu_project/CVAE-GAN/CelebA_DCGAN_results/generator_param.pkl'
# model_dir_D = '/home1/yixu/yixu_project/CVAE-GAN/CelebA_DCGAN_results/discriminator_param.pkl'
#
# if os.path.exists(model_dir_D):
# 	checkpoint_G = torch.load(model_dir_G)
# 	checkpoint_D = torch.load(model_dir_D)
# 	G.load_state_dict(checkpoint_G)
# 	D.load_state_dict(checkpoint_D)
# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999), eps=1e-08, weight_decay=0)
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999), eps=1e-08, weight_decay=0)

# results save folder
if not os.path.isdir('CelebA_DCGAN_results'):
	os.mkdir('CelebA_DCGAN_results')
if not os.path.isdir('CelebA_DCGAN_results/Random_results'):
	os.mkdir('CelebA_DCGAN_results/Random_results')
if not os.path.isdir('CelebA_DCGAN_results/Fixed_results'):
	os.mkdir('CelebA_DCGAN_results/Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

print('Training start!')
start_time = time.time()
for epoch in range(train_epoch):
	D_losses = []
	G_losses = []

	# learning rate decay
	if (epoch + 1) == 11:
		G_optimizer.param_groups[0]['lr'] /= 10
		D_optimizer.param_groups[0]['lr'] /= 10
		print("learning rate change!")

	if (epoch + 1) == 16:
		G_optimizer.param_groups[0]['lr'] /= 10
		D_optimizer.param_groups[0]['lr'] /= 10
		print("learning rate change!")

	num_iter = 0

	epoch_start_time = time.time()
	for batch_idx, x_ in enumerate(train_loader):
		# train discriminator D
		D.zero_grad()

		if isCrop:
			x_ = x_[:, :, 22:86, 22:86]

		mini_batch = x_.size(0)

		y_real_ = torch.ones(mini_batch)
		y_fake_ = torch.zeros(mini_batch)

		x_, y_real_, y_fake_ = x_.cuda(), y_real_.cuda(), y_fake_.cuda()
		# warning
		z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
		z_ = z_.cuda()
		G_result = G(z_)
		if batch_idx%20==0:
			D_result = D(x_)
			D_real_loss = BCE_loss(D_result, y_real_)
			D_result = D(G_result)
			D_fake_loss = BCE_loss(D_result, y_fake_)
			D_fake_score = D_result.data.mean()
			D_train_loss = D_real_loss + D_fake_loss
			D_train_loss.backward()
			D_optimizer.step()
			D_losses.append(D_train_loss.item())

		# train generator G
		G.zero_grad()

		z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
		z_ = z_.cuda()

		G_result = G(z_)
		D_result = D(G_result).squeeze()
		G_train_loss = BCE_loss(D_result, y_real_)
		G_train_loss.backward()
		G_optimizer.step()

		G_losses.append(G_train_loss.item())

		num_iter += 1
		print('[{}/{}]'.format(epoch, train_epoch) +
			  '[{}/{}]'.format(batch_idx, len(train_loader)) +
			  'D_loss:{:g} g_loss:{:g} '.format(D_train_loss, G_train_loss))

	epoch_end_time = time.time()
	per_epoch_ptime = epoch_end_time - epoch_start_time

	print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % (
	(epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
	torch.mean(torch.FloatTensor(G_losses))))
	p = 'CelebA_DCGAN_results/Random_results/CelebA_DCGAN_' + str(epoch + 1) + '.png'
	fixed_p = 'CelebA_DCGAN_results/Fixed_results/CelebA_DCGAN_' + str(epoch + 1) + '.png'
	show_result((epoch + 1), save=True, path=p, isFix=False)
	show_result((epoch + 1), save=True, path=fixed_p, isFix=True)
	train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
	train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
	train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (
torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G.state_dict(), "CelebA_DCGAN_results/generator_param.pkl")
torch.save(D.state_dict(), "CelebA_DCGAN_results/discriminator_param.pkl")
with open('CelebA_DCGAN_results/train_hist.pkl', 'wb') as f:
	pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path='CelebA_DCGAN_results/CelebA_DCGAN_train_hist.png')

images = []
for e in range(train_epoch):
	img_name = 'CelebA_DCGAN_results/Fixed_results/CelebA_DCGAN_' + str(e + 1) + '.png'
	images.append(imageio.imread(img_name))
imageio.mimsave('CelebA_DCGAN_results/generation_animation.gif', images, fps=5)