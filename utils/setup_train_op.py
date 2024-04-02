import os
import shutil
import torch

from datasets_prep.brain_datasets import CreateTrainDataset
from utils.models.discriminator import Discriminator_large
from utils.models.ncsnpp_generator_adagn import NCSNpp
from utils.EMA import EMA

"""
Contains boilerplate code for setting up training pipeline
"""

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))

def set_seed(seed_no):
    torch.manual_seed(seed_no)
    torch.cuda.manual_seed(seed_no)
    torch.cuda.manual_seed_all(seed_no)
    return

def prepare_dataset(rank, args):
    """ Create dataset. Return train sampler, data loader, and args. """
    dataset = CreateTrainDataset()
    train_sampler = torch.utils.data.distributed.DistributedSampler(
                        dataset, num_replicas=args.world_size, rank=rank)
    data_loader = torch.utils.data.DataLoader(
                        dataset, batch_size=args.batch_size, sampler=train_sampler,
                        shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    return train_sampler, data_loader, args

def initialize_models(args, device):
    """Initialize the Generator and Discriminator models."""
    netG = NCSNpp(args).to(device)
    netD = Discriminator_large(nc=2*args.num_channels, ngf=args.ngf,
                               t_emb_dim=args.t_emb_dim, act=torch.nn.LeakyReLU(0.2)).to(device)
    netG = torch.nn.parallel.DistributedDataParallel(netG, device_ids=[device])
    netD = torch.nn.parallel.DistributedDataParallel(netD, device_ids=[device])
    return netG, netD

def setup_optim_schedulers(netG, netD, args):
    """Setup the optimizers and schedulers for both models."""
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    if args.use_ema:
        optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)

    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.num_epoch,
                                                            eta_min=1e-5)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, args.num_epoch,
                                                            eta_min=1e-5)
    return optimizerG, optimizerD, schedulerG, schedulerD

def setup_experiment_directory(exp_path):
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        copy_source(__file__, exp_path)
        shutil.copytree('utils/models', os.path.join(exp_path, 'utils/models'))

def load_checkpoint(exp_path, device, netG, optimizerG, schedulerG, netD, optimizerD, schedulerD):
    checkpoint_file = os.path.join(exp_path, 'content.pth')
    checkpoint = torch.load(checkpoint_file, map_location=device)
    netG.load_state_dict(checkpoint['netG_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG'])
    schedulerG.load_state_dict(checkpoint['schedulerG'])
    netD.load_state_dict(checkpoint['netD_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD'])
    schedulerD.load_state_dict(checkpoint['schedulerD'])
    print(f"* loaded checkpoint (epoch {checkpoint['epoch']})")
    return checkpoint['epoch'], checkpoint['global_step']

def broadcast_params(params):
    for param in params:
        torch.distributed.broadcast(param.data, src=0)
