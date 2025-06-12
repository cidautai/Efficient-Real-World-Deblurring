import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.nn.parallel import DistributedDataParallel as DDP
from ptflops import get_model_complexity_info

from .nafnet import NAFNet, NAFNetLocal  

def create_model(opt, local_rank, global_rank=1):
    '''
    Creates the model.
    opt: a dictionary from the yaml config key network
    '''
    name = opt['name']

    if name == 'NAFNet':
        model = NAFNet(img_channel=opt['img_channels'], 
                        width=opt['width'], 
                        middle_blk_num=opt['middle_blk_num'], 
                        enc_blk_nums=opt['enc_blk_nums'],
                        dec_blk_nums=opt['dec_blk_nums'])#.to(rank)
    elif name == 'NAFNet_Local':
        model = NAFNetLocal(img_channel=opt['img_channels'], 
                width=opt['width'], 
                middle_blk_num=opt['middle_blk_num'], 
                enc_blk_nums=opt['enc_blk_nums'],
                dec_blk_nums=opt['dec_blk_nums'])#.to(rank)

    else:
        raise NotImplementedError('This network is not implemented')
    if global_rank ==0:
        print( '**************************** \n',f'Using {name} network')

        input_size = (3, 1200, 1920)
        macs, params = get_model_complexity_info(model, input_size, print_per_layer_stat = False)
        print(f' ---- Computational complexity at {input_size}: {macs}')
        print(' ---- Number of parameters: ', params)    
    else:
        macs, params = None, None

    model.to(local_rank)

    model = DDP(model, device_ids=[local_rank])
    
    return model, macs, params

def create_optim_scheduler(opt, model):
    '''
    Returns the optim and its scheduler.
    opt: a dictionary of the yaml config file with the train key
    '''
    optim = torch.optim.AdamW( filter(lambda p: p.requires_grad, model.parameters()) , 
                            lr = opt['lr_initial'],
                            weight_decay = opt['weight_decay'],
                            betas = opt['betas'])
    T_max = int(sum(opt['epochs']))

    if opt['lr_scheme'] == 'CosineAnnealing':
        scheduler = CosineAnnealingLR(optim, T_max=T_max, eta_min=opt['eta_min'])
    else: 
        raise NotImplementedError('scheduler not implemented')    
        
    return optim, scheduler

def load_weights(model, model_weights, global_rank = 1):
    '''
    Loads the weights of a pretrained model, picking only the weights that are
    in the new model.
    '''
    new_weights = model.state_dict()
    new_weights.update({k: v for k, v in model_weights.items() if k in new_weights})
    
    model.load_state_dict(new_weights)

    total_checkpoint_keys = len(model_weights)
    total_model_keys = len(new_weights)
    matching_keys = len(set(model_weights.keys()) & set(new_weights.keys()))

    if global_rank==0:
        print(f"Total keys in checkpoint: {total_checkpoint_keys}")
        print(f"Total keys in model state dict: {total_model_keys}")
        print(f"Number of matching keys: {matching_keys}")

    return model

def load_optim(optim, optim_weights):
    '''
    Loads the values of the optimizer picking only the weights that are in the new model.
    '''
    optim.load_state_dict(optim_weights)
    return optim

def resume_model(model,
                 optim,
                 scheduler, 
                 path_model, 
                 local_rank,
                 global_rank,resume:str=None):
    
    '''
    Returns the loaded weights of model and optimizer if resume flag is True
    '''
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    if resume:
        checkpoints = torch.load(path_model, map_location=map_location, weights_only=False)
        weights = checkpoints['model_state_dict']
        model = load_weights(model, model_weights=weights,global_rank = global_rank)

        start_epochs = 0
        if global_rank == 0: print(' ---- Loaded weights', '\n ***************************')
    else:
        start_epochs = 0
        if global_rank==0: print(' ---- Starting from zero the training', '\n ***************************')
    
    return model, optim, scheduler, start_epochs

def save_checkpoint(model, optim, scheduler, metrics_eval, metrics_train, paths, global_rank = 1):

    '''
    Save the .pt of the model after each epoch.
    
    '''
    if global_rank != 0: 
        return None
    
    if type(next(iter(metrics_eval.values()))) != dict:
        metrics_eval = {'metrics': metrics_eval}

    weights = model.state_dict()

    # Save the model after every epoch
    model_to_save = {
        'epoch': metrics_train['epoch'],
        'model_state_dict': weights,
        'optimizer_state_dict': optim.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }

    try:
        torch.save(model_to_save, paths)
        # print(f"Model saved to {paths['new']}")

    except Exception as e:
        print(f"Error saving model: {e}")


__all__ = ['create_model', 'resume_model', 'create_optim_scheduler', 'save_checkpoint',
           'load_optim', 'load_weights']



    
