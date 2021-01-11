import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectralnorm
from torch.nn import init

class Encoder(nn.Module):
	def __init__(self, out_dim, ch):
		super(Encoder, self).__init__()

		self.b1 = nn.Sequential(
			nn.Conv2d(3, ch, 4, 2, 1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2, True)
		)

		self.b2 = nn.Sequential(
			nn.Conv2d(ch, ch*2, 4, 2, 1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2, True)
		)
		
		self.b3 = nn.Sequential(
			nn.Conv2d(ch*2, ch*4, 4, 2, 1),
			nn.BatchNorm2d(ch*4),
			nn.LeakyReLU(0.2, True)
		)
		
		self.b4 = nn.Sequential(
			nn.Conv2d(ch*4, ch*8, 4, 2, 1),
			nn.BatchNorm2d(ch*8),
			nn.LeakyReLU(0.2, True)
		)

		self.b5 = nn.Sequential(
			nn.Conv2d(ch*8, out_dim, 4, 1, 0),
		)
		
		self.init_weights()

	def init_weights(self):
		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				init.normal_(module.weight, 0, 0.02)
			elif isinstance(module, nn.BatchNorm2d):
				init.normal_(module.weight.data, 1.0, 0.02)
				init.constant_(module.bias.data, 0.0)


	def forward(self, x):

		h1 = self.b1(x)
		h2 = self.b2(h1)
		h3 = self.b3(h2)
		h4 = self.b4(h3)
		h5 = self.b5(h4)

		return [h5, h4, h3, h2, h1] 


class DecBlock(nn.Module):
	def __init__(self, in_channel, mid_channel, out_channel, k_s, k_t, s_s, s_t, p_s, p_t, to_rgb=False):
		super(DecBlock, self).__init__()

		self.to_rgb = to_rgb

		if not to_rgb:
			self.conv = nn.Sequential(
				nn.ConvTranspose3d(in_channel, mid_channel, (k_t,1,1), (s_t,1,1), (p_t,0,0)),
				nn.BatchNorm3d(mid_channel),
				nn.LeakyReLU(0.2, True),
				nn.ConvTranspose3d(mid_channel, out_channel, (1,k_s,k_s), (1,s_s,s_s), (0,p_s,p_s)),
				nn.BatchNorm3d(out_channel),
				nn.LeakyReLU(0.2, True)
			)
		else:
			self.conv = nn.Sequential(
				nn.ConvTranspose3d(in_channel, mid_channel, (k_t,1,1), (s_t,1,1), (p_t,0,0)),
				nn.BatchNorm3d(mid_channel),
				nn.LeakyReLU(0.2, True),
				nn.ConvTranspose3d(mid_channel, out_channel, (1,k_s,k_s), (1,s_s,s_s), (0,p_s,p_s)),
				nn.Tanh()
			)

	def forward(self, h, hm=None, ha=None):

		if self.to_rgb:
			h = self.conv(h)
		else:
			h = self.conv(h)
			ha_r = ha.unsqueeze(2).repeat(1, 1, h.size(2), 1, 1)
			hm_r = hm.unsqueeze(-1).repeat(1, 1, h.size(2), h.size(3), h.size(4))
			h = torch.cat([h, ha_r, hm_r], dim=1)

		return h


class Decoder(nn.Module):
	def __init__(self, c_z=512, c_a=100, c_m=6, ch=64):
		super(Decoder, self).__init__()

		self.block1 = DecBlock(c_z+c_a+c_m, 4096, ch*8, 4, 2, 1, 1, 0, 0) # 4 x 4 x 4
		self.block2 = DecBlock(ch*8*2+c_m,  3072, ch*4, 4, 4, 2, 2, 1, 1) # 8 x 8 x 8
		self.block3 = DecBlock(ch*4*2+c_m,  1536, ch*2, 4, 4, 2, 2, 1, 1) # 16 x 16 x 16
		self.block4 = DecBlock(ch*2*2+c_m,   768,	ch, 4, 4, 2, 2, 1, 1) # 16 x 32 x 32
		self.to_rgb = DecBlock(ch*2+c_m,	  36,	 3, 4, 4, 2, 2, 1, 1, True) # 16 x 64 x 64

		self.init_weights()

	def init_weights(self):
		for module in self.modules():
			if isinstance(module, nn.Conv3d):
				init.normal_(module.weight, 0, 0.02)
			elif isinstance(module, nn.BatchNorm3d):
				init.normal_(module.weight.data, 1.0, 0.02)
				init.constant_(module.bias.data, 0.0)

	def forward(self, z, c, feats):

		f0 = feats[0]		
		
		z = z.unsqueeze(-1).unsqueeze(-1)
		c = c.unsqueeze(-1).unsqueeze(-1)
		h = torch.cat([z, c, f0], 1).unsqueeze(-1)

		h = self.block1(h, c, feats[1])
		h = self.block2(h, c, feats[2])
		h = self.block3(h, c, feats[3])
		h = self.block4(h, c, feats[4])
		out = self.to_rgb(h)

		return out


class Generator(nn.Module):
	def __init__(self, c_z=512, c_a=100, c_m=6, ch=64):
		super(Generator, self).__init__()

		self.enc = Encoder(c_a, ch)
		self.dec = Decoder(c_z, c_a, c_m, ch)

	def forward(self, x, z, c):
		
		feats = self.enc(x)
		out = self.dec(z, c, feats)
		
		return out


class VideoDiscriminator(nn.Module):
	def __init__(self, nclasses=0, ch=64):
		super(VideoDiscriminator, self).__init__()

		self.nclasses = nclasses

		self.net1 = nn.Sequential(
			spectralnorm(nn.Conv3d(3,	ch, 4, 2, 1)),
			nn.LeakyReLU(0.2, inplace=True),
		)

		self.net2 = nn.Sequential(
			spectralnorm(nn.Conv3d(ch+nclasses, ch*2, 4, 2, 1)),
			nn.LeakyReLU(0.2, inplace=True),

			spectralnorm(nn.Conv3d(ch*2, ch*4, 4, 2, 1)),
			nn.LeakyReLU(0.2, inplace=True),

			spectralnorm(nn.Conv3d(ch*4, ch*8, 4, 2, 1)),
			nn.LeakyReLU(0.2, inplace=True),

			spectralnorm(nn.Conv3d(ch*8, 1, (2,4,4), 1, 0)),
		)

		self.init_weights()

	def init_weights(self):
		for module in self.modules():
			if isinstance(module, nn.Conv3d):
				init.normal_(module.weight, 0, 0.02)

	def forward(self, x, cm=None):
	
		h = self.net1(x)		
	
		if cm is not None:
			cm = cm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
			cm_r = cm.repeat(1, 1, h.size(2), h.size(3), h.size(4))
			h = torch.cat([h, cm_r], dim=1)	
		
		out = self.net2(h)		

		return out.squeeze(-1).squeeze(-1).squeeze(-1).squeeze(-1)


class ImageDiscriminator(nn.Module):
	def __init__(self, ch=64):
		super(ImageDiscriminator, self).__init__()

		self.net = nn.Sequential(
			spectralnorm(nn.Conv2d(3, ch, 4, 2, 1)),
			nn.LeakyReLU(0.2, inplace=True),

			spectralnorm(nn.Conv2d(ch, ch*2, 4, 2, 1)),
			nn.LeakyReLU(0.2, inplace=True),

			spectralnorm(nn.Conv2d(ch*2, ch*4, 4, 2, 1)),
			nn.LeakyReLU(0.2, inplace=True),

			spectralnorm(nn.Conv2d(ch*4, ch*8, 4, 2, 1)),
			nn.LeakyReLU(0.2, inplace=True),

			spectralnorm(nn.Conv2d(ch*8, 1, 4, 1, 0)),
		)

		self.init_weights()

	def init_weights(self):
		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				init.normal_(module.weight, 0, 0.02)

	def forward(self, x):

		out = self.net(x)

		return out.squeeze(-1).squeeze(-1).squeeze(-1)


if __name__ == '__main__':

	net = Generator()
	x = torch.randn(4, 3, 64, 64)
	z = torch.randn(4, 512)
	c = torch.randn(4, 6)
	out = net(x, z, c)
	print(out.size())

	vdis = VideoDiscriminator(c_m=6)
	out = vdis(out, c)	
	print(out.size())

