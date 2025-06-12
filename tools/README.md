In this submodule the main functions for optimizing and testing the model are defined. It has been decided to define this out of the [train.py](/train.py) scripts to reduce the code load in this file.

## Trainer

The main function defined is `train_model`. It follows a simple optimization pipeline: a main loop of training, through all the batches in the train dataset split. Something very relevant of this training loop is the use of Distributed Data Parallel. It is important to load correctly the batches into the correspondent device.

The results from the forward method of the network must be in the form of a dictionary (check [/archs/README.md](/archs/README.md)) and will be introduced directly into the `calculate_loss` function --- the function in charge of calculating all the optimizations based in the selected losses in the config file (check [/losses/README.md](/losses/README.md)).

## Tester

There are two main functions defined `eval_one_loader` and `eval_model`. The last is the one that is called during `train.py` or `test.py`, and as can be seen in the code, it will call `eval_one_loader` for each element of the test loader defined (check [/data/README.md](/data/README.md) for more info in test loader construction).  
The `eval_one_loader` will make the inference of all the test loader batches and save information of the metrics into a list. Finally the mean of all the batches is performed to return a performance evaluation.

> **NOTE**. The test pipeline proposed in this repository is not the one that will be used in the submission page. The one that will be used resembles [evaluate_RSBlur.py](https://github.com/rimchang/RSBlur/blob/main/evaluation/evaluate_RSBlur.py) and it is not considered in the metrics calculation of the repository due to its large computation time.

