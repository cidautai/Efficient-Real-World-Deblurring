## How to create a model

In this folder (`archs`), the architecture of the networks that you would like to run must be defined. The `__init__.py` file works as the connection of the contents of `archs` folder with all the framework. Thus, you may define a network in a given file (`network.py`) and import it in `_init__.py` like

```python
from .network import Network
```

After having it imported you need to add the network name into the if/else statements in `create_model` function. This function is the one that it is called during the training to instantiate the model. Just write:

```python
def create_model(opt):

...
...

elif name == 'Network':
    model = Network(*args)
```

When finished these simple steps, by selecting the name of the network in the config file, it will be automatically instantiated.

## Structure of the network (nn.Module)

The network that you want to train must be defined in a .py file, namely `Network.py`. This python class must be inherited from a `nn.Module`. The output of the network is given by the `forward` method and must be a python dictionary with all the values that you want to return:

```python
def forward(self, inp, train=False):

    ...
    ...
    if train:
        return {'output': output, 'value1': value1, 'value2': value2, ...}
    else:
        return {'output': output}
```

> Important! Even if you want to return only the output value, it must be contained in a python dictionary. Otherwise, the framework will return an error. Also, it is important to label the deblurred final output with the 'output' key, as the framework expects this value to be the inferred image. 

## Check network instantiation is working

Finally, to debug in a modular way, you may find interesting to inferred an image with your defined model. You can follow the [NAFNet](/archs/nafnet.py) exampleand check the `if __name__ == '__main__'` statement. Using `ptflops` you can make an inference of an image through the network, while checking the model parameters and MACs. This can also help you control the computational cost of the method.

```python
if __name__ == '__main__':
    from ptflops import get_model_complexity_info

    net = Network(*args)

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    print(macs, params)

```

## Optimizer and Scheduler

Optimizer and Scheduler are both instantiated with the `create_optim_scheduler` function in `__init__.py`. You may change the optimizer to any other that you want, currently the used one is AdamW. 

The Scheduler is setted to be defined based on the config file `lr_scheme` key. If you want to use other scheduler than the ones defined in the if/else statement, then you must import in the `__init__.py` the desired scheduler and add the elif naming to it in the `create_optim_scheduler` function (following the `create_model` steps). 