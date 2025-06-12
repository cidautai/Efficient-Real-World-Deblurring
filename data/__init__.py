
from .dataset_tools.dataset_RSBlur import main_dataset_rsblur

# from .create_data import create_data

def shuffle_sampler(samplers, epoch):
    '''
    A function that shuffles all the Distributed samplers in the loaders.
    '''
    if not samplers: # if they are none
        return
    for sampler in samplers:
        sampler.set_epoch(epoch)

def adjust_list_length(lst, target_length):
    """
    Adjusts the length of a list to match the target length by either appending
    the last element or removing elements from the end.

    Parameters:
        lst (list): The list to adjust.
        target_length (int): The desired length of the list.

    Returns:
        list: The adjusted list with the specified length.
    """
    assert isinstance(lst, list), "First argument must be a list."
    assert isinstance(target_length, int), "Second argument must be an integer."
    assert target_length >= 0, "Target length must be non-negative."

    while len(lst) < target_length:
        lst.append(lst[-1] if lst else None)
    while len(lst) > target_length:
        lst.pop()
    return lst

def check_if_defined(opt, key, value, step=0):
    '''
    Check if a key is defined in the provided options dictionary and return its corresponding value.

    Parameters:
        opt (dict): The dictionary containing options.
        key (str): The key to check in the dictionary.
        value (Any): The default value to return if the key is not found in the dictionary.
        step (int, optional): The index to use if the value associated with the key is a list. Defaults to 0.

    Returns:
        Any: The value associated with the key in the dictionary if it exists and is not a list, 
             or the value at the specified index if it is a list. If the key is not found, 
             the provided default value is returned.
    '''
    if key not in opt:
        return value
    elif key in opt and type(opt[key]) != list:
        return opt[key]
    else:
        return adjust_list_length(opt[key], step+1)[step]   # we adjust the length of the list to be the same as the number of steps (prevent errors in the yaml file)


def create_data(rank, world_size, opt, step = 0):
    '''
    opt: a dictionary from the yaml config key datasets 
    '''
    name = opt['name']
    train_path=opt['train']['train_path']
    test_path = opt['val']['test_path']
    verbose = opt['train']['verbose']
    batch_size_test = opt['val']['batch_size_test']
    
    # the following parameters are only defined in the train set
    batch_size_train = check_if_defined(opt['train'], 'batch_size_train', world_size, step=step) // world_size
    flips = check_if_defined(opt['train'], 'flips', False,step=0)
    cropsize = check_if_defined(opt['train'], 'cropsize', None, step=step)
    num_workers = check_if_defined(opt['train'], 'num_workers', 1, step=0)
    crop_type = check_if_defined(opt['train'], 'crop_type', 'Random', step=0)
    
    
    if rank != 0:
        verbose = False
    samplers = None # TEmporal change!!

    if name == 'RSBlur':
        train_loader, test_loader, samplers = main_dataset_rsblur(rank=rank,
                                                train_path=train_path,
                                                batch_size_train=batch_size_train,
                                                test_path = test_path,
                                                batch_size_test=batch_size_test,
                                                verbose=verbose,
                                                num_workers=num_workers,
                                                crop_type=crop_type,
                                                cropsize=cropsize,
                                                flips=flips,
                                                world_size=world_size)

    else:
        raise NotImplementedError(f'{name} is not implemented')        
    # print(samplers, train_loader, test_loader)
    if rank ==0: 
        print(f'Using {name} Dataset')
        print(f'Crop Size of {cropsize}')
        print(f'Batch Size of {batch_size_train} per GPU', '\n ****************************')
    
    return train_loader, test_loader, samplers

__all__ = ['create_data', 'shuffle_sampler']
