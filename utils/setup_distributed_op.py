import os
import torch

def setup_distributed_environment(rank, size, master_addr, master_port):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    torch.distributed.init_process_group(backend='nccl', init_method='env://',
                                         rank=rank, world_size=size)

def cleanup_distributed_environment():
    """ Cleanup the distributed environment. """
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
