import torch
import random
from pytorch_msssim import ssim
from  metrics import psnr

def train(args, epoch, G, VD, ID, optimizer_G, optimizer_VD, optimizer_ID, criterion_gan, criterion_l1, dataloader, writer, device):

	# train mode
	G.train()
	VD.train()
	ID.train()

	for i, (vid, cls) in enumerate(dataloader):

		global_steps = epoch * len(dataloader) + i
		bs = vid.size(0)
		real_vid = vid.to(device)
		real_start_frame = vid[:,:,0,:,:]
		real_img = real_vid[:,:,random.randint(0, vid.size(2)-1), :, :]
		cls = cls.to(device)		

		#################### train G ##################
		optimizer_G.zero_grad()

		z = torch.randn(bs, args.dim_z).to(device)

		fake_vid = G(real_start_frame, z, cls)
		fake_img = fake_vid[:,:, random.randint(0, vid.size(2)-1),:,:]		

		# recon loss
		err_recon = criterion_l1(fake_vid, real_vid)
		
		# gan loss
		VG_fake = VD(fake_vid, cls)
		IG_fake = ID(fake_img)

		y_real = torch.ones(VG_fake.size()).to(device)
		y_fake = torch.zeros(VG_fake.size()).to(device)

		errVG = criterion_gan(VG_fake, y_real)
		errIG = criterion_gan(IG_fake, y_real)

		# total loss
		errG = args.weight_l1 * err_recon + errVG + errIG

		errG.backward()
		optimizer_G.step()

		#################### train D ##################
		optimizer_VD.zero_grad()
		optimizer_ID.zero_grad()

		VD_real = VD(real_vid, cls)
		ID_real = ID(real_img)

		VD_fake = VD(fake_vid.detach(), cls)
		ID_fake = ID(fake_img.detach())

		errVD = criterion_gan(VD_real, y_real) + criterion_gan(VD_fake, y_fake)
		errID = criterion_gan(ID_real, y_real) + criterion_gan(ID_fake, y_fake)
		
		errVD.backward()
		optimizer_VD.step()

		errID.backward()
		optimizer_ID.step()
	
		# logging	
		writer.add_scalar('G_vid_recon', err_recon.item(), global_steps)
		writer.add_scalar('G_vid_loss', errVG.item(), global_steps)
		writer.add_scalar('G_img_loss', errIG.item(), global_steps)
		writer.add_scalar('D_vid_loss', errVD.item(), global_steps)
		writer.add_scalar('D_img_loss', errID.item(), global_steps)
		writer.flush()

		if global_steps % args.print_freq == 0:
			print("[Epoch %d/%d] [Iter %d/%d] [Recon: %f] [VD loss: %f] [VG loss: %f] [ID loss: %f] [IG loss: %f]"
				  %(epoch, args.max_epoch, i, global_steps, err_recon.item(), errVD.item(), errVG.item(), errID.item(), errIG.item()))


def val(args, epoch, G, criterion_l1, dataloader, device, writer):

	with torch.no_grad():
		
		G.eval()

		l1_loss = []
		ssim_loss = []
		psnr_loss = []		

		for i, (vid, cls) in enumerate(dataloader):

			vid = vid.to(device)
			img = vid[:,:,0,:,:]
			cls = cls.to(device)
			
			bs = vid.size(0)
			z = torch.randn(bs, args.dim_z).to(device)			

			vid_recon = G(img, z, cls)
			
			# l1 loss
			err_l1 = criterion_l1(vid_recon, vid)
			l1_loss.append(err_l1)			

			vid = vid.transpose(2,1).contiguous().view(-1, 3, 64, 64)
			vid_recon = vid_recon.transpose(2,1).contiguous().view(-1,3,64,64)

			# ssim
			vid = (vid + 1) / 2 # [0, 1]
			vid_recon = (vid_recon + 1) / 2 # [0, 1]
			err_ssim = ssim(vid, vid_recon, data_range=1, size_average=False)
			ssim_loss.append(err_ssim.mean().item())			
	
			# psnr
			err_psnr = psnr(vid, vid_recon)			
			psnr_loss.append(err_psnr.mean().item())

		l1_avg = sum(l1_loss) / len(l1_loss)
		ssim_avg = sum(ssim_loss) / len(ssim_loss)
		psnr_avg = sum(psnr_loss) / len(psnr_loss)

		writer.add_scalar('val/l1_recon', l1_avg, epoch)
		writer.add_scalar('val/ssim', ssim_avg, epoch)
		writer.add_scalar('val/psnr', psnr_avg, epoch)
		writer.flush()

		print("[Epoch %d/%d] [l1: %f] [ssim: %f] [psnr: %f]"%(epoch, args.max_epoch, l1_avg, ssim_avg, psnr_avg))


def test(args, epoch, G, dataloader, device, writer):
	
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

			writer.add_video(tag='vid_cat_%d'%i, global_step=epoch, vid_tensor=vid_gen)
			writer.flush()
			
