import os

# PyTorch library
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms

try:
    from .datapipeline import Crop_Dataset
    from .utils import separate_elements, check_paths
except:
    from datapipeline import Crop_Dataset
    from utils import separate_elements, check_paths

def separate_elements(file_path):
    list1 = []
    list2 = []

    with open(file_path, 'r') as file:
        for line in file:
            elements = line.strip().split()
            if len(elements) == 2:
                list1.append(elements[0])
                list2.append(elements[1])

    return list1, list2

def main_dataset_rsblur(rank = 0,
                        train_path='./data/RSBlur',
                         batch_size_train=4,
                         test_path='./data/RSBlur',
                         batch_size_test=1, 
                         verbose=False, 
                         num_workers=1, 
                         crop_type='Random',
                         cropsize=256,
                         flips=True,
                         world_size = 1):

    PATH_TRAIN = os.path.join(test_path, 'RSBlur_real_train.txt')

    PATH_VALID = os.path.join(test_path, 'RSBlur_real_test.txt')

    paths_sharp_train, paths_blur_train = separate_elements(PATH_TRAIN)
    paths_sharp_valid, paths_blur_valid = separate_elements(PATH_VALID)

    
    list_blur_train = [os.path.join(train_path, path) for path in paths_blur_train]
    list_sharp_train = [os.path.join(train_path, path) for path in paths_sharp_train]

    list_blur_valid = [os.path.join(test_path, path) for path in paths_blur_valid]
    list_sharp_valid = [os.path.join(test_path, path) for path in paths_sharp_valid]     

    # check if all the image routes are correct
    check_paths([list_blur_train, list_sharp_train, list_blur_valid, list_sharp_valid])

    if verbose:
        print('Images in the subsets: \n')
        print("    -Images in the PATH_LOW_TRAIN folder: ", len(list_blur_train))
        print("    -Images in the PATH_HIGH_TRAIN folder: ", len(list_sharp_train))
        print("    -Images in the PATH_LOW_VALID folder: ", len(list_blur_valid))
        print("    -Images in the PATH_HIGH_VALID folder: ", len(list_sharp_valid))

    tensor_transform = transforms.ToTensor()
    if flips:
        flip_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # flips horizontal with p=0.5
            transforms.RandomVerticalFlip()  # flips vertical with p = 0.5
        ])
    else:
        flip_transform = None

    # Load the datasets
    train_dataset = Crop_Dataset(list_blur_train, list_sharp_train, cropsize=cropsize,
                                  tensor_transform=tensor_transform, flips=flip_transform,test=False, 
                                  crop_type=crop_type)
    test_dataset = Crop_Dataset(list_blur_valid, list_sharp_valid, cropsize=None,
                                  tensor_transform=tensor_transform, test=True, 
                                  crop_type=crop_type)

    # Now we need to apply the Distributed sampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, shuffle=True, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, shuffle= True, rank=rank)

    samplers = []
    # samplers = {'train': train_sampler, 'test': [test_sampler_gopro, test_sampler_lolblur]}
    samplers.append(train_sampler)
    samplers.append(test_sampler)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False, sampler=test_sampler)

    test_loaders = {'RSBlur': test_loader}

    return train_loader, test_loaders, samplers

if __name__ == '__main__':
    
    PATH_TEST = '/media/Data/Datasets_Dani/RSBlur'

    train_loader, test_loader, samplers = main_dataset_rsblur(train_path=PATH_TEST,test_path=PATH_TEST, verbose=True)

    for high, low in test_loader['RSBlur']:
        print(high.shape, low.shape)