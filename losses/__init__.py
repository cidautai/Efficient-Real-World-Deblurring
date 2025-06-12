from .loss import PixelLoss, VGGLoss, EdgeLoss, SSIM

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


def create_loss(opt, local_rank, step = 0, global_rank=1):
    """
    Creates and returns a dictionary of loss functions and their corresponding weights 
    for evaluating a model based on the provided options.
    This function dynamically initializes various loss functions based on the configuration 
    specified in the `opt` dictionary for a given training step. It supports multiple types 
    of losses such as pixel loss, perceptual loss, edge loss, edge alignment loss, frequency 
    loss, enhance loss, YUV loss, and color consistency loss (or every loss that you defined 
    previously in loss.py and import to this script). Each loss is initialized with 
    its respective parameters and moved to the specified device rank.
    Args:
        opt (dict): A dictionary containing configuration options for the losses. 
                    It includes keys for each loss type (e.g., 'pixel', 'perceptual', etc.) 
                    and their respective parameters such as weights, criteria, and reductions.
        rank (int): The device rank (e.g., GPU ID) where the losses should be moved.
        step (int, optional): The current training step, used to index into step-specific 
                                configurations in the `opt` dictionary. Defaults to 0.
    Returns:
        dict: A dictionary containing the initialized loss functions, where the keys are 
                the names of the losses (e.g., 'pixel_loss', 'perceptual_loss') and the values 
                are the corresponding loss objects.
    Note: 
        The flags to each loss are processed by the adjust_list_length function, which ensures that
        the length of the list is equal to the number of steps. This prevents errors in the YAML file
        if the user does not define the same number of weights for each step. The function will
        automatically adjust the length of the list to be equal to the number of steps.
    """
    losses = dict()
    weight_losses = dict()

    # these are the keywords that need to be checked if have the proper length
    processing_keys = ['flag', 'weight']

    # process all the keys in the optimization dictionary and adjust their length if needed
    for key, value in opt.items():
        if key in processing_keys:
            opt[key] = adjust_list_length(value, opt['STEPS'])

    # first the pixel losses
    if opt['pixel_flag'][step]: 
        pixel_loss = PixelLoss(loss_weight = opt['pixel_weight'][step],
                                criterion = opt['pixel_criterion'],
                                reduction = opt['pixel_reduction']).to(local_rank)
        losses['pixel_loss'] = pixel_loss
        weight_losses['pixel_weight'] = opt['pixel_weight'][step]
        if global_rank == 0: print(f"Using pixel loss {opt['pixel_criterion']} with weight {opt['pixel_weight'][step]}")
    
    # now the perceptual loss
    if opt['perceptual_flag'][step]:     
        perceptual_loss = VGGLoss(loss_weight = opt['perceptual_weight'][step],
                                criterion = opt['perceptual_criterion'],
                                reduction = opt['perceptual_reduction']).to(local_rank)
        losses['perceptual_loss'] = perceptual_loss
        weight_losses['perceptual_weight'] = opt['perceptual_weight'][step]
        if global_rank == 0: print(f"Using perceptual loss {opt['perceptual_criterion']} with weight {opt['perceptual_weight'][step]}")

    # the edge loss
    if opt['edge_flag'][step]: 
        edge_loss = EdgeLoss(loss_weight = opt['edge_weight'][step],
                                criterion = opt['edge_criterion'],
                                reduction = opt['edge_reduction'],
                                rank = local_rank).to(local_rank)
        losses['edge_loss'] = edge_loss
        weight_losses['edge_weight'] = opt['edge_weight'][step]
        if global_rank == 0: print(f"Using edge loss {opt['edge_criterion']} with weight {opt['edge_weight'][step]}")

    if global_rank==0: 
        print('Used losses: ',losses.keys())
        print(f'With weights {weight_losses.values()}','***************************')
    return losses


def calculate_loss(all_losses,
                   output_dict,
                   high_batch,
                   get_individual_losses = False):
    """
    Calculates and returns the total optimization loss and optionally individual losses.

    This function computes various loss components based on the provided loss functions 
    and input data. It aggregates these losses into a total optimization loss, which 
    can be used for model training. Optionally, it can also return a dictionary of 
    individual loss values.

    Args:
        all_losses (dict): A dictionary containing loss functions. The keys are the 
            names of the losses (e.g., 'pixel_loss', 'perceptual_loss'), and the values 
            are the corresponding callable loss functions.
        output_dict (dict): A dictionary containing the model's output data. It may 
            include keys such as 'output', 'side_align', 'out_side', and 'side_darkir', 
            depending on the losses being calculated.
        high_batch (torch.Tensor): The ground truth high-resolution batch used for 
            calculating the losses.
        get_individual_losses (bool, optional): If True, the function returns a 
            dictionary of individual loss values in addition to the total optimization 
            loss. Defaults to False.
        scale_factor (int, optional): A scaling factor used for certain loss functions 
            (e.g., 'enhance_loss' and 'yuv_loss'). Defaults to 8.

    Returns:
        float: The total optimization loss, which is the sum of all calculated losses.
        dict (optional): A dictionary of individual loss values, where the keys are 
            the loss names and the values are the corresponding loss values. This is 
            returned only if `get_individual_losses` is True.
    """

    # Create a list to store loss computations
    losses = dict()
    optim_loss = 0

    if 'pixel_loss' in all_losses:
        l_pixel = all_losses['pixel_loss'](output_dict['output'], high_batch)
        losses['pixel_loss'] = l_pixel
        optim_loss += l_pixel

    if 'perceptual_loss' in all_losses:
        l_perceptual = all_losses['perceptual_loss'](output_dict['output'], high_batch)
        losses['perceptual_loss'] = l_perceptual
        optim_loss += l_perceptual

    if 'edge_loss' in all_losses:
        l_edge = all_losses['edge_loss'](output_dict['output'], high_batch)
        losses['edge_loss'] = l_edge
        optim_loss += l_edge

    if get_individual_losses:
        return optim_loss, losses
    else:
        return optim_loss




__all__ = ['create_loss', 'calculate_loss', 'SSIM', 'VGGLoss']
