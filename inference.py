import os
from PIL import Image
from options.options import parse
import argparse

parser = argparse.ArgumentParser(description="Script for prediction")
parser.add_argument('-p', '--config', type=str, default='./options/inference/RSBlur.yml', help = 'Config file of prediction')
parser.add_argument('-c', '--checkpoints_path', type=str, default='./models/NAFNet_RSBlur.pt', 
                help="Checkpoints path")
parser.add_argument('-i', '--inp_path', type=str, default='./images/inputs', 
                help="Folder path")
args = parser.parse_args()


path_options = args.config
opt = parse(path_options)
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

# PyTorch library
import torch
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from torchvision.transforms import Resize

import torchvision.transforms as transforms
from archs import create_model

#define some auxiliary functions
pil_to_tensor = transforms.ToTensor()
tensor_to_pil = transforms.ToPILImage()

def path_to_tensor(path):
    img = Image.open(path).convert('RGB')
    img = pil_to_tensor(img).unsqueeze(0)
    
    return img
def normalize_tensor(tensor):
    
    max_value = torch.max(tensor)
    min_value = torch.min(tensor)
    output = (tensor - min_value)/(max_value)
    return output

def save_tensor(tensor, path):
    
    tensor = tensor.squeeze(0)
    print(tensor.shape, tensor.dtype, torch.max(tensor), torch.min(tensor))
    img = tensor_to_pil(tensor)
    img.save(path)

def pad_tensor(tensor, multiple = 16):
    '''pad the tensor to be multiple of some number'''
    multiple = multiple
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value = 0)
    
    return tensor

def load_model(local_rank, model, path_weights, global_rank=1):
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    checkpoints = torch.load(path_weights, map_location=map_location, weights_only=False)

    weights = checkpoints['model_state_dict']
    # weights = {'module.' + key: value for key, value in weights.items()}

    if global_rank == 0 and 'epochs' in checkpoints.keys(): 
        print('Values on epoch:', checkpoints['epoch'])
    model.load_state_dict(weights)
    if global_rank == 0: print('Loaded weights correctly')
    return model

#parameters for saving model
resize = opt['Resize']
scale = opt['scale']

def predict_folder():
    
    dist.init_process_group(backend='nccl')

    # Get distributed settings
    global_rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])

    # DEFINE NETWORK, SCHEDULER AND OPTIMIZER
    model, _, _ = create_model(opt['network'], local_rank=local_rank, global_rank=global_rank)

    model = load_model(local_rank, model, path_weights = args.checkpoints_path, global_rank=global_rank)
    # create data
    PATH_IMAGES= args.inp_path
    PATH_RESULTS = opt['images']['results_path']

    #create folder if it doen't exist
    not os.path.isdir(PATH_RESULTS) and os.mkdir(PATH_RESULTS)

    path_images = [os.path.join(PATH_IMAGES, path) for path in os.listdir(PATH_IMAGES) if path.endswith(('.png', '.PNG', '.jpg', '.JPEG'))]
    path_images = [file for file in path_images if not file.endswith('.csv') and not file.endswith('.txt')]
   
    model.eval()
    if global_rank==0:
        pbar = tqdm(total = len(path_images))
        
    for path_img in path_images:
        tensor = path_to_tensor(path_img).to(local_rank)
        _, _, H, W = tensor.shape
        
        if resize and (H >=1500 or W>=1500):
            new_size = [int(dim//scale) for dim in (H, W)]
            downsample = Resize(new_size)
        else:
            downsample = torch.nn.Identity()
        tensor = downsample(tensor)
        
        tensor = pad_tensor(tensor)

        with torch.no_grad():
            output_dict = model.module(tensor)
            output = output_dict['output']
            # output = output[-1]
        if resize:
            upsample = Resize((H, W))
        else: upsample = torch.nn.Identity()
        output = upsample(output)
        output = torch.clamp(output, 0., 1.)
        # print(output.shape)
        output = output[:,:, :H, :W]
        save_tensor(output, os.path.join(PATH_RESULTS, os.path.basename(path_img)))

        if global_rank == 0:
            pbar.update(1)

    print('Finished inference!')
    if global_rank == 0:
        pbar.close()   
    dist.destroy_process_group()


if __name__ == '__main__':
    predict_folder()