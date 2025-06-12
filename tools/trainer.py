import sys

sys.path.append('../losses')
sys.path.append('../data/datasets/datapipeline')
from losses import SSIM, calculate_loss

calc_SSIM = SSIM(data_range=1.)

def train_model(model, optim, all_losses, train_loader, metrics, local_rank = None):
    '''
    It trains the model, returning the model, optim, scheduler and metrics dict
    '''
    for high_batch, low_batch in train_loader:

        # Move the data to the GPU
        high_batch = high_batch.to(local_rank)
        low_batch = low_batch.to(local_rank)

        optim.zero_grad()
        # Feed the data into the model
        output_dict = model(low_batch)
        
        optim_loss, _ = calculate_loss(all_losses, output_dict, high_batch, get_individual_losses=True)

        optim_loss.backward()
        optim.step()

    return model, optim, metrics