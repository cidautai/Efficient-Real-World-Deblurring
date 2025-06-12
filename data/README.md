## How to schedule the training in a new dataset

As in the other submodules, the [__init__.py](/data/__init__.py) is the file that connects the code in `/data` folder with the training schedule. The main function of this subfolder is the `create_data` one, that will build your train and test loaders given a set of configuration keys. 

In the `/datasets_tools` you can find the scripts that build all of these dataloaders and data readers.  In [datapipeline](/data/dataset_tools/datapipeline.py), you define your `Dataset` inherited data reader. Then in another script, namely `new_dataloader.py`, you may create a function that instantiates the dataloaders. During this step, it is important to return the following objects in the return of this function:

```python
def new_dataloader(*args, world_size, rank):

    ...

    train_dataset = NewDataset(*args)
    test_dataset = NewDataset(*args)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, shuffle= True, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size,shuffle=True,rank=rank)

    samplers = []
    samplers.append(train_sampler)
    samplers.append(test_sampler)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=True, sampler = train_sampler)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=False,
                             num_workers=num_workers, pin_memory=True, drop_last=False, sampler=test_sampler)
    test_loaders = {'New_Dataset': test_loader}

    return train_loader, test_loaders, samplers 
```

The samplers in the function are needed to distribute different batches of images to each GPU (and not have redundant calculation) and they are handled internally by PyTorch.

Finally, you need to import this `new_dataloader` in the `__init__.py` file and add the new Dataset (or combination of Datasets) to the `create_data` function if/else statement.

> In the default state of the repository, the file that will be called is [dataset_RSBlur.py](/data/dataset_tools/dataset_RSBlur.py).