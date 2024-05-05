import torch
import os

from utils.models.ncsnpp_generator_adagn import NCSNpp
from datasets_prep.brain_datasets import CreateDatasetReconstruction

"""
Contains boilerplate code for setting up inference pipeline
"""
def load_checkpoint(checkpoint_file, netG, device, trained = True, epoch_sel = False, epoch = 500):
    checkpoint = torch.load(checkpoint_file, map_location=device)        
    if epoch_sel:
        ckpt = checkpoint 
    else:    
        ckpt = checkpoint['netG_dict']
    if trained:
        for key in list(ckpt.keys()):       
            ckpt[key[7:]] = ckpt.pop(key)
    netG.load_state_dict(ckpt)
    netG.eval()
    return netG

def get_checkpoint_file_name(args):
    if args.epoch_sel:
        checkpoint_file = f'../diffusion_test/{args.dataset}/{args.exp}/netG_{args.epoch_id}.pth'
    else:
        checkpoint_file = f'../diffusion_test/{args.dataset}/{args.exp}/content.pth'
    return checkpoint_file

def load_pretrained_model(args, device, init=False, netG=None):
    if init:
        netG = NCSNpp(args).to(device)
    checkpoint_file = get_checkpoint_file_name(args)
    if args.epoch_sel:
        netG = load_checkpoint(checkpoint_file, netG, device=device, \
                               epoch_sel=True, epoch=args.epoch_id)
    else:
        netG = load_checkpoint(checkpoint_file, netG, device=device)
    return netG

def load_data(args):
    dataset = CreateDatasetReconstruction(phase=args.phase, contrast=args.contrast,\
                                          data=args.which_data, R=args.R)
    data_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.batch_size,
                                               shuffle=args.shuffle,
                                               num_workers=4)
    return data_loader

def set_lr_schedule(args, optimizerG):
    if args.lr_schedule:
        if args.schedule=='cosine_anneal':
            schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.itr_inf, eta_min=1e-5)
        elif args.schedule=='onecycle':
            schedulerG = torch.optim.lr_scheduler.OneCycleLR(optimizerG, max_lr=args.lr_g , total_steps=args.itr_inf)
    else:
        schedulerG = None
    return schedulerG

def set_save_dir(args):
    save_dir = f"../diffusion_test/{args.dataset}/{args.exp}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    #if weights are shared for slices within a subject
    save_dir = save_dir + args.extra_string 
    print(f'save_dir: {save_dir}')
    return save_dir