import pandas as pd
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import glob
import torch

class MUG(Dataset):
	def __init__(self, split, data_path, transform=None):

		self.data_root = data_path
		
		if split == 'train':
			self.anno_path = 'annotation/train_none_neutral.csv'
		elif split == 'val':
			self.anno_path = 'annotation/val_none_neutral.csv'
		else:
			raise NotImplementedError

		self.df = pd.read_csv(self.anno_path)

		self.transform = transform
		self.length = 32
		self.cat_dic = {'sadness':0, 'anger':1, 'surprise':2, 'disgust':3, 'happiness':4, 'fear':5}

	def __getitem__(self, idx):

		video_path = self.df.iloc[idx]['video']
		nframes = int(self.df.iloc[idx]['nframes'])
		cat_code = int(self.cat_dic[self.df.iloc[idx]['category']])

		start_idx = random.randint(0, nframes-32)

		clip = [Image.open(self.data_root + video_path + '/img_%04d.jpg'%(start_idx + i)).convert("RGB") for i in range(self.length)]

		if self.transform is not None:
			vid = self.transform(clip)		

		y = torch.zeros(6)
		y[cat_code] = 1.0		

		return vid, y

	def __len__(self):

		return len(self.df)	

class MUG_test(Dataset):
	def __init__(self, data_path, transform=None):

		self.data_root = data_path
		
		self.anno_path = 'annotation/val_neutral.csv'
		self.df = pd.read_csv(self.anno_path)
		self.transform = transform

	def __getitem__(self, idx):

		video_path = self.df.iloc[idx]['video']

		start_idx = 0
		img = Image.open(self.data_root + video_path + '/img_%04d.jpg'%start_idx).convert("RGB")

		if self.transform is not None:
			img = self.transform(img)		

		return img

	def __len__(self):

		return len(self.df)	
