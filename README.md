# [AIM 2025] Efficient Real-World Deblurring Challenge
<table>
    <tr>
        <td width="20%">
            <img src="https://www.cvlai.net/aim/2025/logos/AIM2020_logo.png" alt="Left Image" style="width:20%;"/>
        </td>
    </tr>
</table>

## Introduction

This is the official repository of the [**Efficient Real-World Deblurring Challenge**](https://www.codabench.org/competitions/8483/#/pages-tab) at [**AIM 2025**](https://www.cvlai.net/aim/2025/). The computational cost pipeline that will be followed by the organizers is proposed in it. In addition a training baseline framework for the methods is included. Three different tasks can be performed in the repository: 
- train -> [train.sh](/train.sh)    ([train.py](/train.py)).
- inference -> [inference.sh](/inference.sh) ([inference.py](/inference.py)).
- computing cost -> [cost.sh](/cost.sh) ([computing_cost.py](/computing_cost.py)).

Each of these scripts are sketched as the executors of functions defined in other submodules --- such as `/archs`, `/options`, `/data`, etc. The configuration files for some of these tasks are included in the `/options` folder.

## Installation

This repository is tested to work with CUDA 12.4, Python 3.10.12 and PyTorch >= 2.3.1. Begin by cloning the repository:

```
git clone https://github.com/danifei/Efficient-Real-World-Deblurring.git
```

For the installation of the libraries run

```
python -m venv venv_rsblur
source venv_rsblur/bin/activate
pip install - requirements.txt
```

## Dataset

The dataset used for this challenge is [RSBlur](https://github.com/rimchang/RSBlur). You can download the `RSBLUR.zip` from [Google Drive](https://drive.google.com/drive/folders/1sS8_qXvF4KstJtyYN1DDHsqKke8qwgnT). The dataset splits are the following ones:

- [Training set](https://github.com/rimchang/RSBlur/blob/main/datalist/RSBlur/RSBlur_real_train.txt).
- [Validation set](https://github.com/rimchang/RSBlur/blob/main/datalist/RSBlur/RSBlur_real_val.txt).
- [Test set](https://github.com/rimchang/RSBlur/blob/main/datalist/RSBlur/RSBlur_real_test.txt).

> The Validation and Test sets cannot be used to train. They are only expected to be used for performance comprobations. **Solutions that overfit the public validation and test splits will be disqualified**.

This repository expects the RSBlur dataset to be put in the [/data/datasets](/data/datasets). Next, the dataset splits text documents are expected to be in `/data/datasets/RSBlur/`. After this, the datasets paths are configured to work with the training pipeline of this repository.

The development input images can be downloaded from [development_input_RSBlur.zip](https://drive.google.com/file/d/18t4EhyJDc4QnHpaV2viu6ZagV5wS-jiN/view?usp=sharing).

## Scripts

The scripts that need to be run to use the train, inference and computation cost calculations are [train.sh](/train.sh), [inference.sh](/inference.sh) and [cost.sh](/cost.sh), respectively. To run each of their corresponding python scripts (such as [train.py](/train.py) for [train.sh](/train.sh)) the torchrun command is used. This command easily lets us define a Distributed Data Parallel (DDP) environment. For training this environment is especially important as it allows the use of various GPUs to accelerate the training or work with larger batch sizes.

## Inference

For the evaluation of your solutions, first you will need to generate the deblurred results list of images. You can use the proposed inference pipeline for this task. Configure the input files and output files in [inference.sh](/inference.sh). 

Then, in [/options/inference/RSBlur.yml](/options/inference/RSBlur.yml) set the network architecture that will be called. Note that you need to define your architecture in [/archs](/archs/) following [/archs/README.md](/archs/README.md). Finally run:

```
sh inference.sh
```

Next, generate the zip file that contains at its root the deblurred results. Finally upload the .zip file to the competition **submission section**.

> Note that the deblurred results must match in name with their respective input images. This should be handled by this repository, but checking is recommended.


## Training

The workflow intended in this repository to train is:

1. Define the [datasets](/data/README.md), [archs](/archs/README.md) and [losses](/losses/README.md). Also check if the optimizer and scheduler defined in `create_optimizer_scheduler` ([archs](/archs/__init__.py)) are the ones needed. Default is AdamW and Cosine annealing, respectively.

2. Check that the trainer and tester follow the training steps that you would expect (check [/tools/README.md](/tools/README.md)).

3. If you have made significant changes to the [trainer](/tools/trainer.py), it is recommended to review the `train.py` script. While these scripts are designed to work with the current schedules, modifications may lead to unexpected results.

4. Now you are able to modify the [configuration file](/options/train/RSBlur.yml) based on the training pipeline that you want to perform.

5. Finally you will run `sh train.sh`. Check the arguments that need to be configured in the proper script.

## Computational Cost

A computation calculation pipeline is also included in this repository. This will give the participants an intuition around the computational cost of their methods and if they meet the restrictions criteria:

- 5 million parameters
- 200 GMACs

Begin by configuring the [Runtime.yml](/options/test/Runtime.yml) with your method architecture. Then run

```
sh cost.sh
```
> This script will also return an inference runtime of the models. You may add this runtime to the **readme.txt** in your zip file submission. You may also specify which GPU/CPU you were using. 

## Acknowledgement
The proposed baseline is based in the original [NAFNet](https://github.com/megvii-research/NAFNet) implementation.

## Contact

If you have any doubt, contact [danfei@cidaut.es](danfei@cidaut.es).
