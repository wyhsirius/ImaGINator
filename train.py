from __future__ import absolute_import

import torch
import torch.nn as nn
from trainer import train, val, test
from model.networks import Generator, VideoDiscriminator, ImageDiscriminator
import torchvision
import transforms_vid
from dataset import UVA, MUG, MUG_test
from torch.utils.tensorboard import SummaryWriter
import os
import cfg
from torchvision import transforms

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main():

	args = cfg.parse_args()
	torch.cuda.manual_seed(args.random_seed)

	print(args)

	# create logging folder
	log_path = os.path.join(args.save_path, args.exp_name + '/log')
	model_path = os.path.join(args.save_path, args.exp_name + '/models')
	os.makedirs(log_path, exist_ok=True)
	os.makedirs(model_path, exist_ok=True)
	writer = SummaryWriter(log_path) # tensorboard

	# load model
	print('==> loading models')
	device = torch.device("cuda:0")

	G = Generator(args.dim_z, args.dim_a, args.nclasses, args.ch).to(device)
	VD = VideoDiscriminator(args.nclasses, args.ch).to(device)
	ID = ImageDiscriminator(args.ch).to(device)

	G = nn.DataParallel(G)
	VD = nn.DataParallel(VD)
	ID = nn.DataParallel(ID)

	# optimizer
	optimizer_G = torch.optim.Adam(G.parameters(), args.g_lr, (0.5, 0.999))
	optimizer_VD = torch.optim.Adam(VD.parameters(), args.d_lr, (0.5, 0.999))
	optimizer_ID = torch.optim.Adam(ID.parameters(), args.d_lr, (0.5, 0.999))

	# loss
	criterion_gan = nn.BCEWithLogitsLoss().to(device)
	criterion_l1 = nn.L1Loss().to(device)

	# prepare dataset
	print('==> preparing dataset')
	transform = torchvision.transforms.Compose([
		transforms_vid.ClipResize((args.img_size, args.img_size)),
		transforms_vid.ClipToTensor(),
		transforms_vid.ClipNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
	)

	transform_test = torchvision.transforms.Compose([
		transforms.Resize((args.img_size, args.img_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
	)

	if args.dataset == 'mug':
		dataset_train = MUG('train', args.data_path, transform=transform)
		dataset_val = MUG('val', args.data_path, transform=transform)
		dataset_test = MUG_test(args.data_path, transform=transform_test)
	else:
		raise NotImplementedError

	dataloader_train = torch.utils.data.DataLoader(
		dataset = dataset_train,
		batch_size = args.batch_size,
		num_workers = args.num_workers,
		shuffle = True,
		pin_memory = True,
		drop_last = True
	)
	
	dataloader_val = torch.utils.data.DataLoader(
		dataset = dataset_val,
		batch_size = args.batch_size,
		num_workers = args.num_workers,
		shuffle = False,
		pin_memory = True
	)

	dataloader_test = torch.utils.data.DataLoader(
		dataset = dataset_test,
		batch_size = args.batch_size_test,
		num_workers = args.num_workers,
		shuffle = False,
		pin_memory = True
	)

	print('==> start training')
	for epoch in range(args.max_epoch):
		train(args, epoch, G, VD, ID, optimizer_G, optimizer_VD, optimizer_ID, criterion_gan, criterion_l1, dataloader_train, writer, device)
		
		if epoch % args.val_freq == 0:
			val(args, epoch, G, criterion_l1, dataloader_val, device, writer)
			test(args, epoch, G, dataloader_test, device, writer)			

		if epoch % args.save_freq == 0:
			torch.save(G.state_dict(), os.path.join(model_path, 'G_%d.pth'%(epoch)))
			torch.save(VD.state_dict(), os.path.join(model_path, 'VD_%d.pth'%(epoch)))
			torch.save(ID.state_dict(), os.path.join(model_path, 'ID_%d.pth'%(epoch)))

	return

if __name__ == '__main__':

	main()
