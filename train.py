
import os
import time
from tqdm import tqdm
from options.options import parse
import random
import argparse

parser = argparse.ArgumentParser(description="Script for train")
parser.add_argument('-p', '--config', type=str, default='./options/train/RSBlur.yml', help = 'Config file of prediction')
args = parser.parse_args()

opt = parse(args.config)

import torch
import torch.distributed as dist

from data.dataset_tools.datapipeline import *
from archs import *
from losses import *
from data import *
from utils.utils import create_path_models, set_random_seed
from tools.trainer import train_model
from tools.tester import eval_model

torch.autograd.set_detect_anomaly(True)

#parameters for saving model
PATH_RESUME, PATH_SAVE = create_path_models(opt['save'])

final_score = 0.

def run_model():
    # Initialize distributed environment
    dist.init_process_group(backend='nccl')
    
    # Get distributed settings
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    
    # Set device
    torch.cuda.set_device(local_rank)
    
    print(f'Rank: {global_rank} of {world_size}, Local rank: {local_rank}')

    seed = opt['datasets']['seed']
    if seed is None:
        seed = random.randint(1, 10000)
        opt['datasets']['seed'] = seed
    set_random_seed(seed + global_rank)

    # DEFINE NETWORK, SCHEDULER AND OPTIMIZER
    model, macs, params = create_model(opt['network'], local_rank=local_rank, global_rank=global_rank)

    # save this stats into opt 
    opt['macs'] = macs
    opt['params'] = params
    opt['Total_GPUs'] = world_size # add the number of GPUs to the opt

    # define the optimizer
    optim, scheduler = create_optim_scheduler(opt['train'], model)

    # if resume load the weights
    model, optim, scheduler, _ = resume_model(model, optim, scheduler, path_model = PATH_RESUME,
                                                         local_rank = local_rank, global_rank = global_rank, 
                                                         resume=opt['network']['resume_training'])

    # last_epochs = start_epochs
    for step in range(opt['train']['STEPS']):
        total_steps = opt['train']['STEPS']
        if global_rank == 0: print(f'--------------- In Step {step + 1} of {total_steps} Steps ---------------')
        # LOAD THE DATALOADERS
        train_loader, test_loader, samplers = create_data(global_rank, world_size=world_size, opt = opt['datasets'], step = step)

        # create losses in this step
        all_losses = create_loss(opt['train'], step = step,local_rank=local_rank, global_rank=global_rank)
        final_score= 0
        
        if global_rank==0:
            total = opt['train']['epochs'][step]
            pbar = tqdm(total = total)
        for epoch in range(opt['train']['epochs'][step]):

            start_time = time.time()
            metrics_train = {'epoch': epoch,'final_score': final_score}
            metrics_eval = {}

            # shuffle the samplers of each loader
            shuffle_sampler(samplers, epoch)

            # train phase
            model.train()
            model, optim, metrics_train = train_model(model, optim, all_losses, train_loader, metrics_train,local_rank = local_rank)

            # eval phase
            if epoch % opt['train']['eval_freq'] == 0 or epoch == opt['train']['epochs'][step] - 1:
                model.eval()
                metrics_eval = eval_model(model, test_loader, metrics_eval, local_rank=local_rank, world_size=world_size)
                
                # print some results
                if global_rank==0:
                    print(f"Epoch {epoch + 1} of {opt['train']['epochs'][step]} took {time.time() - start_time:.3f}s\n")
                    if type(next(iter(metrics_eval.values()))) == dict:
                        for key, metric_eval in metrics_eval.items():
                            print(f" \t {key} --- PSNR: {metric_eval['valid_psnr']}, SSIM: {metric_eval['valid_ssim']}, LPIPS: {metric_eval['valid_lpips']}")
                    else:
                        print(f" \t {opt['datasets']['name']} --- PSNR: {metrics_eval['valid_psnr']}, SSIM: {metrics_eval['valid_ssim']}, LPIPS: {metrics_eval['valid_lpips']}")
                    # update progress bar
                    pbar.update(1)
            # Save the model after every epoch
                final_score = save_checkpoint(model, optim, scheduler, metrics_eval = metrics_eval, metrics_train=metrics_train, 
                                        paths = PATH_SAVE,global_rank=global_rank)

            #update scheduler
            scheduler.step()
        if global_rank == 0:
            pbar.close()

if __name__ == '__main__':
    run_model()

