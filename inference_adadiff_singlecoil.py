import argparse
import itertools
import torchvision
import torch
import numpy as np

import torch.nn.functional as F
import torchvision.transforms as transforms

from train_adadiff_singlecoil import Posterior_Coefficients, sample_posterior
from utils.args_op import add_singlecoil_inference_args
from utils.setup_inference_op import load_pretrained_model, load_data, set_lr_schedule, \
                                     set_save_dir

MAX_ITER = 20

def to_range_0_1(x):
    return (x + 1.) / 2.

def div_mean(x):
    return x/x.mean()

def psnr(img1, img2):
    """
    Compute Peak Signal to Noise Ratio value between two images.
    """
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(img1.max() / torch.sqrt(mse))

def crop_us(us, x):
    """
    Crop x according to dimensions of us

    This is needed to crop to the original dimensions, since loaded data are 256x256
    which doesn't correspond to the original dimensions
    """
    return transforms.CenterCrop((us.shape[-2],us.shape[-1]))(x)

#%% posterior sampling
def sample_from_model(coefficients, generator, n_time, us, mask, args, device):
    """
    Commented code corresponds to how I understood the paper, but it somehow
    degrades the quality of the reconstructions (metric: PSNR)
    """
    x_breve = -1 * torch.ones(args.batch_size, args.num_channels, args.image_size, args.image_size).to(device)

    x_dot = data_consistency(x_breve, us, mask)
    x = x_dot # First iteration: x_dot is better than x_breve
    with torch.no_grad():
        n_batch = x_breve.size(0)
        for i in reversed(range(n_time)):
            t = torch.full((n_batch,), i, dtype=torch.int64).to(device)
            latent_z = torch.randn(n_batch, args.nz, device=device)

            # x0_tilda = generator(x_dot, t, latent_z)
            # x_breve = sample_posterior(coefficients, x0_tilda, x_dot, t).detach()
            # x_dot = data_consistency(x_breve, us, mask)
            x_0 = generator(x, t, latent_z)
            x_0 = data_consistency(x_0, us, mask)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()

        # x = x_breve.detach()
        x_0 = generator(x, t, latent_z)
        x = x_0.detach()
    return x

def data_consistency(x, us, mask, range_adj = True, reshape = True):
    mask=mask>0.5
    pad_y = int ( ( x.shape[-2]-us.shape[-2])/2 )
    pad_x = int ( ( x.shape[-1]-us.shape[-1])/2 )
    pad = torch.nn.ZeroPad2d((pad_x, pad_x, pad_y, pad_y))
    x = x * 0.5 + 0.5
    if reshape:
        x = crop_us(us, x)
    x = fft2c(x) * ~mask + fft2c(us) * mask
    x = torch.abs(ifft2c(x))
    if reshape:
        x = pad(x)
    if range_adj:    
        x = ( x - 0.5 ) / 0.5
    return x

def data_consistency_loss(x, us, mask):
    mask=mask>0.5
    x = x * 0.5 + 0.5
    if x.shape[-1] != us.shape[-1]:
        x = crop_us(us, x)
        x_fft = fft2c(x) * mask
        us_fft =  fft2c(us) * mask                
        loss = F.l1_loss(x_fft, us_fft)
    return loss

def ifft2c(x, dim=((-2,-1)), img_shape=None):
    x = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=dim), s=img_shape, dim=dim), dim = dim)
    return x

def fft2c(x, dim=((-2,-1)), img_shape=None):
    x = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=dim), s=img_shape, dim=dim), dim = dim)
    return x

def prepare_data(fs, us, acs, mask, acs_mask, device):
    #make us complex
    us = us[:,[0],:]*np.exp(1j*(us[:,[1],:]*2*np.pi-np.pi))
    # make acs complex
    acs = acs[:,[0],:]*np.exp(1j*(acs[:,[1],:]*2*np.pi-np.pi))
    #move variables to device
    us = us.to(device)
    fs = fs.to(device)
    acs = acs.to(device)
    mask = mask.to(device)
    acs_mask = acs_mask > 0.5 # make boolean
    acs_mask = acs_mask.to(device)
    fs = crop_us(us, fs)
    return fs, us, acs, mask, acs_mask

def apply_data_consistency_ACS(synth_image_res, acs, acs_mask):
    """
    input: synth_image_res (256x152 image), acs (256x152 image), acs_mask (256x152 binary mask)
    output: consist_image (256x152 image)
    function: add [fourier transform of synth_image_res] and [fourier transform acs where acs_mask is 1], then take inverse fourier transform
    """
    consist_frequency = fft2c(synth_image_res) * ~acs_mask + fft2c(acs) * acs_mask
    consist_image = torch.abs(ifft2c(consist_frequency))
    return consist_image

def apply_data_consistency_ACS_256(synth_image_res, acs, acs_mask):
    """
    input: synth_image_res (256x256 image), acs (256x152 image), acs_mask (256x152 binary mask)
    output: consist_image (256x256 image)
    function: add [fourier transform of synth_image_res] and [fourier transform acs where acs_mask is 1], then take inverse fourier transform
    * pad 0's to acs_mask to make it 256x256
    """
    acs = torch.nn.functional.pad(acs, (52, 52, 0, 0))
    acs_mask = torch.nn.functional.pad(acs_mask, (52, 52, 0, 0))
    consist_frequency = fft2c(synth_image_res) * ~acs_mask + fft2c(acs) * acs_mask
    consist_image = torch.abs(ifft2c(consist_frequency))
    return consist_image

def save_recon_results(save_dir, args, recons, recons_inter, psnr_res, synth_image_res, us, fs, image_idx):
    np.save(f'{save_dir}{args.contrast}_{args.phase}_{args.R}_recons_{args.itr_inf}_final.npy', recons)
    if args.save_inter:
        np.save(f'{save_dir}{args.contrast}_{args.phase}_{args.R}_recons_{args.itr_inf}_inter.npy', recons_inter)

    np.save(f'{save_dir}{args.contrast}_{args.phase}_{args.R}_psnr_{args.itr_inf}_final.npy', psnr_res)

    synth_final_res = to_range_0_1(crop_us(us, synth_image_res))
    fs = to_range_0_1(fs)
    us = torch.abs(us)
    print(f'image_idx - {image_idx}')
    print(f'PSNR {psnr(div_mean(synth_final_res), div_mean(fs))}')

    torchvision.utils.save_image(torch.cat((us, synth_final_res, fs), axis=-1),
                                    f'{save_dir}{args.contrast}_{args.phase}_{args.R}_samples_{image_idx}.jpg',
                                    normalize=True)
    return

def sample_and_test(args):
    torch.manual_seed(42)
    device = torch.device('cuda:{}'.format(args.local_rank))

    data_loader = load_data(args)
    shape = data_loader.dataset[0][1].shape

    netG = load_pretrained_model(args, device, init=True)
    args2 = args
    args2.exp = "hfs_0505"
    netG_HFS = load_pretrained_model(args2, device, init=True)

    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    schedulerG = set_lr_schedule(args, optimizerG)
    pos_coeff = Posterior_Coefficients(args, device)
    save_dir = set_save_dir(args)

    psnr_res = np.zeros((MAX_ITER, args.itr_inf))
    recons = np.zeros((MAX_ITER, shape[-2], shape[-1]))

    # if intermediate recons (during adaptation) also needed to be saved
    if args.save_inter:
        recons_inter = np.zeros((int(args.itr_inf/100+1), MAX_ITER, shape[-2], shape[-1]), \
                                 dtype = np.float32)

    for image_idx, (fs, us, acs, mask, acs_mask) in enumerate(data_loader):
        if image_idx > MAX_ITER:
            break
        fs, us, acs, mask, acs_mask = prepare_data(fs, us, acs, mask, acs_mask, device)

        # Rapid Diffusion
        rapid_diffusion_res = sample_from_model(pos_coeff, netG_HFS, args.num_timesteps, us, mask, args, device)
        rapid_diffusion_res = apply_data_consistency_ACS_256(rapid_diffusion_res, acs, acs_mask)

        # Prior Adaptation
        t = torch.zeros([1], device=device)
        latent_z = torch.randn(1, args.nz, device=device)
        for ii in range(args.itr_inf):
            netG.zero_grad()    
            synth_image = netG(rapid_diffusion_res, t, latent_z)
            lossDC = data_consistency_loss(synth_image, us, mask)

            # apply data consistency
            synth_image = crop_us(us, data_consistency(synth_image, us, mask))
            synth_image[fs==-1] = -1

            psnr_res[image_idx, ii] = psnr(div_mean(to_range_0_1(crop_us(us, synth_image))), div_mean(to_range_0_1(fs)))
            # # apply data consistency w.r.t. ACS
            # synth_image_res = apply_data_consistency_ACS(synth_image, acs, acs_mask)

            # psnr_res[image_idx, ii] = psnr(div_mean(to_range_0_1(crop_us(us, synth_image_res))), div_mean(to_range_0_1(fs)))

            # backward pass
            lossDC.backward() 
            optimizerG.step()
            if args.lr_schedule:
                schedulerG.step()
            synth_image = synth_image.detach()
            # save intermediate reconstruction
            if ii % 100 == 0 and args.save_inter:
                # recons_inter[int(ii/100), image_idx, :] = np.squeeze(to_range_0_1(crop_us(us, synth_image_res)).cpu().detach().numpy())
                recons_inter[int(ii/100), image_idx, :] = np.squeeze(to_range_0_1(crop_us(us, synth_image)).cpu().numpy())

        # save final reconstruction
        # recons[image_idx, :] = np.squeeze(to_range_0_1(crop_us(us, synth_image_res)).cpu().detach().numpy())
        # save_recon_results(save_dir, args, recons, recons_inter, psnr_res, synth_image_res, us, fs, image_idx)
        recons[image_idx, :] = np.squeeze(to_range_0_1(crop_us(us, synth_image)).cpu().numpy())
        save_recon_results(save_dir, args, recons, recons_inter, psnr_res, synth_image, us, fs, image_idx)

        # reset generator
        if args.reset_opt:            
            optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))        
            if args.lr_schedule:
                if args.schedule=='cosine_anneal':
                    print(args.schedule)
                    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.itr_inf, eta_min=1e-5)
                elif args.schedule=='onecycle':
                    print(args.schedule)
                    schedulerG = torch.optim.lr_scheduler.OneCycleLR(optimizerG, max_lr = args.lr_g , total_steps = args.itr_inf)
        netG = load_pretrained_model(args, device, netG=netG)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('adadiff parameters')
    parser = add_singlecoil_inference_args(parser)
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    sample_and_test(args)
