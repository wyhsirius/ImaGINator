from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from model.networks import Generator
import skvideo.io
import numpy as np
import os
import torchvision
from dataset import MUG_test
from torchvision import transforms
import argparse


def save_videos(path, vids, n, cat):

	for i in range(n):
		v = (vids[i].permute(0, 2, 3, 1) * 255).to(torch.uint8)
		torchvision.io.write_video(os.path.join(path, "%d_%d.mp4" % (i, cat)), v, fps=24.0)
		#skvideo.io.vwrite(os.path.join(path, "%d_%d.mp4" % (i, cat)), v, outputdict={"-vcodec": "libx264"})

	return


def main(args):

	# write into tensorboard
	log_path = os.path.join('demos', args.dataset + '/log')
	vid_path = os.path.join('demos', args.dataset + '/vids')

	os.makedirs(log_path, exist_ok=True)
	os.makedirs(vid_path, exist_ok=True)
	writer = SummaryWriter(log_path)

	device = torch.device("cuda:0")

	G = Generator(args.dim_z, args.dim_a, args.nclasses, args.ch).to(device)
	G = nn.DataParallel(G)
	G.load_state_dict(torch.load(args.model_path))

	transform = torchvision.transforms.Compose([
		transforms.Resize((args.img_size, args.img_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
	)

	dataset = MUG_test(args.data_path, transform=transform)

	dataloader = torch.utils.data.DataLoader(
		dataset = dataset,
		batch_size = args.batch_size,
		num_workers = args.num_workers,
		shuffle = False,
		pin_memory = True
	)

	with torch.no_grad():

		G.eval()

		img = next(iter(dataloader))

		bs = img.size(0)
		nclasses = args.nclasses

		z = torch.randn(bs, args.dim_z).to(device)

		for i in range(nclasses):
			y = torch.zeros(bs, nclasses).to(device)
			y[:,i] = 1.0
			vid_gen = G(img, z, y)

			vid_gen = vid_gen.transpose(2,1)
			vid_gen = ((vid_gen - vid_gen.min()) / (vid_gen.max() - vid_gen.min())).data

			writer.add_video(tag='vid_cat_%d'%i, vid_tensor=vid_gen)
			writer.flush()

			# save videos
			print('==> saving videos')
			save_videos(vid_path, vid_gen, bs, i)


if __name__ == '__main__':

	parser = argparse.ArgumentParser('imaginator demo config')

	parser.add_argument('--dataset', type=str, default='mug')
	parser.add_argument('--data_path', type=str, default='')
	parser.add_argument('--img_size', type=int, default=64)
	parser.add_argument('--batch_size', type=int, default=10)
	parser.add_argument('--num_workers', type=int, default=8)
	parser.add_argument('--dim_z', type=int, default=512)
	parser.add_argument('--dim_a', type=int, default=100)
	parser.add_argument('--nclasses', type=int, default=6)
	parser.add_argument('--ch', type=int, default=64)
	parser.add_argument('--model_path', type=str, default='')
	parser.add_argument('--random_seed', type=int, default='12345')

	args = parser.parse_args()

	main(args)
