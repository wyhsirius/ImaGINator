import argparse


def parse_args():

	parser = argparse.ArgumentParser('imaginator training config')

	# train
	parser.add_argument('--max_epoch', type=int, default=5001, help='number of epochs of training')
	parser.add_argument('--batch_size', type=int, default=64, help='train batch size')
	parser.add_argument('--batch_size_test', type=int, default=10, help='test batch size')
	parser.add_argument('--g_lr', type=float, default=2e-4, help='learning rate of generator')
	parser.add_argument('--d_lr', type=float, default=2e-4, help='learning rate of discriminator')
	parser.add_argument('--weight_l1', type=float, default=100, help='weight l1')
	parser.add_argument('--dim_z', type=int, default=512, help='z dim')
	parser.add_argument('--dim_a', type=int, default=100, help='appearance dim')
	parser.add_argument('--nclasses', type=int, default=6, help='category number')
	parser.add_argument('--num_workers', type=int, default=16, help='number of workers')
	parser.add_argument('--ch', type=int, default=64, help='base channels')
	parser.add_argument('--img_size', type=int, default=64, help='generate image size')
	parser.add_argument('--dataset', type=str, default='mug', choices=['mug', 'uva'], help='dataset choice')
	parser.add_argument('--val_freq', type=int, default=50, help='validation frequence')
	parser.add_argument('--print_freq', type=int, default=100, help='log frequence')
	parser.add_argument('--save_freq', type=int, default=100, help='model save frequence')
	parser.add_argument('--exp_name', type=str, default='v1')
	parser.add_argument('--save_path', type=str, default='./exps', help='model and log save path')
	parser.add_argument('--data_path', type=str, default='', help='dataset path')
	parser.add_argument('--random_seed', type=int, default='12345')

	args = parser.parse_args()

	return args
