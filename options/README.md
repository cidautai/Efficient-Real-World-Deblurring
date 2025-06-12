## Configuration files

This subfolder contains all the configuration files needed for training, inferencing and cost computation of your methods. The whole repository is intended to work around these configuration files. 

> You can have as many keys written in the configs, that they will only be used the ones needed to work. For example, if you want to add a new network architecture, you'll surely need to add new keys to define them at [/archs/__init__.py](/archs/__init__.py). The existing keys don't need to be removed if they are unused. The requested keys will be handled automatically.

## Train configurations

The train configurations may follow the structure in [train](/options/train/RSBlur.yml). The keys are divided into 4 different topics:

<!-- - The **device** information, that gathers the most important information on the environment training setup. -->

- The **dataset** section, where information related to the used dataset for training is requested.

- **network** section stablishes the arguments of the network architecture. It also decides if you resume training.

- **optimization** section is the one related to the whole optimization process. Information about optimizer, scheduler, epochs, steps and used losses (per step) is introduced here.

- **save** section states where to save the checkpoints of the model. This will only be performed after testset evaluation.

> Note that in the **dataset** and **optimization** sections, there are a few keys that need to receive a list of element (whether they are bools, ints, floats or lists). They are marked with a commented **STEP**. All these elements that need a list are related with a posible multi-step training that can be performed within this framework. The `optimization.steps` key is the one that stablishes how many stages in the training will be used. If you set `optimization.steps=1`, then only the first element of the mention keys will be used. Please pay attention to the len of these key lists (the ones with a comment of `STEP`), that should be the same as the number of steps for every of them. If only one step is performed, it has been tested that the framework will work properly if the elements of this keys are lists, e.g. `optimization.pixel_flag: [True]`. Otherwise, It won't work.

> There are some keys that also need to be introduced as lists, but don't have the `STEP` comment on them. They won't change through the stages.

## Inference and Computation Cost configurations

These two types of config files have fewer number of keys, as all the training don't need to be done. In the case of inference additional `Resize` and `scale` keys are added to handle large scale images.

