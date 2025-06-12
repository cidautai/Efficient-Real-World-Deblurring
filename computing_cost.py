import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.distributed as dist
import numpy as np

from archs import *

from options.options import parse
import argparse

parser = argparse.ArgumentParser(description='Test the runtime of the model')
parser.add_argument('-p', '--config', type=str, default='./options/cost/Runtime.yml', help='config file')
args = parser.parse_args()
opt = parse(args.config)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def compute_time():
    
    dist.init_process_group(backend='nccl')

    # Get distributed settings
    global_rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
    model, _, _ = create_model(opt['network'], local_rank, global_rank=global_rank)

   
    size = 1000
    mean_time = []
    print("Start timing ...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    for i in range(size):
        
        img = torch.randn((1, 3, 1200, 1920)).to(device)

        with torch.no_grad():
            start.record()
            output = model(img)
            end.record()
            torch.cuda.synchronize()
        mean_time.append(start.elapsed_time(end))
        
        del(output)
        torch.cuda.empty_cache()
        if i%100 == 0:
            print('iteration:', i)

    
    print('Estimated computing time was:', np.mean(mean_time), 'ms')
    dist.destroy_process_group()

if __name__ == '__main__':
    compute_time()