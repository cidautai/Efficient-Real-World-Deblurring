import torch
import sys
from lpips import LPIPS
import numpy as np
sys.path.append('../losses')
sys.path.append('../data/datasets/datapipeline')
from losses.loss import SSIM

import torch.distributed as dist

calc_SSIM = SSIM(data_range=1.)

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def eval_one_loader(model, test_loader, metrics, local_rank=0, world_size = 1):
    calc_LPIPS = LPIPS(net = 'vgg', verbose=False).to(local_rank)
    mean_metrics = {'valid_psnr':[], 'valid_ssim':[], 'valid_lpips':[]}
    with torch.no_grad():
        # Now we need to go over the test_loader and evaluate the results of the epoch
        for high_batch_valid, low_batch_valid in test_loader:

            
            high_batch_valid = high_batch_valid.to(local_rank)
            low_batch_valid = low_batch_valid.to(local_rank)          

            output_dict = model(low_batch_valid)

            # loss
            valid_loss_batch = torch.mean((high_batch_valid - output_dict['output'])**2)
            valid_ssim_batch = calc_SSIM(output_dict['output'], high_batch_valid)
            valid_lpips_batch = calc_LPIPS(output_dict['output'], high_batch_valid)
                
            valid_psnr_batch = 20 * torch.log10(1. / torch.sqrt(valid_loss_batch))        
    
            mean_metrics['valid_psnr'].append(valid_psnr_batch.item())
            mean_metrics['valid_ssim'].append(valid_ssim_batch.item())
            mean_metrics['valid_lpips'].append(torch.mean(valid_lpips_batch).item())

    valid_psnr_tensor = reduce_tensor(torch.tensor(np.mean(mean_metrics['valid_psnr'])).to(local_rank), world_size=world_size)
    valid_ssim_tensor = reduce_tensor(torch.tensor(np.mean(mean_metrics['valid_ssim'])).to(local_rank),world_size=world_size)
    valid_lpips_tensor = reduce_tensor(torch.tensor(np.mean(mean_metrics['valid_lpips'])).to(local_rank), world_size=world_size)

    metrics['valid_psnr'] = valid_psnr_tensor.item()
    metrics['valid_ssim'] = valid_ssim_tensor.item()
    metrics['valid_lpips'] = valid_lpips_tensor.item()
    
    return metrics


def eval_model(model, test_loader, metrics, local_rank=None, world_size = 1):
    '''
    This function runs over the multiple test loaders and returns the whole metrics.
    '''

    #first you need to assert that test_loader is a dictionary
    if type(test_loader) != dict:
        test_loader = {'data': test_loader}

    all_metrics = {}

    for key, loader in test_loader.items():
        all_metrics[f'{key}'] = {}
        metrics = eval_one_loader(model, loader, all_metrics[f'{key}'], local_rank=local_rank, world_size=world_size)
        all_metrics[f'{key}'] = metrics
    
    return all_metrics
