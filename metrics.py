import torch

def psnr(img1, img2):
	
	img1 = img1 * 255
	img2 = img2 * 255
	mse = torch.mean((img1 - img2) ** 2)
	
	return 20 * torch.log10(255.0 / torch.sqrt(mse))


