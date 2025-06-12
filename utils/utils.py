import os
import numpy as np
import torch
import random
import torch.distributed as dist

def setup(rank, world_size, port = '12355'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    # print('before starting the process')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
def cleanup():
    dist.destroy_process_group()

def create_path_models(opt):
    
    PATH_MODEL     = opt['path_resume'] 
    NEW_PATH_MODEL = opt['path_save']

    return PATH_MODEL, NEW_PATH_MODEL

def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    
    pass
