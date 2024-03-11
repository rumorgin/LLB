# Rethinking Few-Shot Class-Incremental Learning: A Lazy Learning Baseline

PyTorch implementation of Rethinking Few-Shot Class-Incremental Learning: A Lazy Learning Baseline

## Abstract

Few-shot class-incremental learning is a step forward in the realm of incremental learning, catering to a more 
realistic context. In typical incremental learning scenarios, the initial session possesses ample data for 
effective training. However, subsequent sessions often lack sufficient data, leading the model to simultaneously 
face the challenges of catastrophic forgetting in incremental learning and overfitting in few-shot learning. 
Existing methods employ fine-tuning strategy on new session to carefully maintain a balance of plasticity 
and stability. In this study, we challenge this balance and design a lazy learning baseline that is more biased 
towards stability: pre-training a feature extractor with initial session data and fine-tuning a cosine classifier. 
For new sessions, we forgo further training and instead use class prototypes for classification. Experiments 
across CIFAR100, miniImageNet, and CUB200 benchmarks reveal our approach outperforms state-of-the-art methods. 
Furthermore, detailed analysis experiments uncover a common challenge in existing few-shot class-incremental 
learning: the low accuracy of new session classes. We provide insightful explanations for these challenges. 
Finally, we introduce a new indicator, separate accuracy, designed to more accurately describe the performance 
of methods in handling both old and new classes.

## Requirements

tqdm~=4.64.1
numpy~=1.23.5
pillow~=9.3.0
torchvision~=0.15.0

## Datasets and pretrained models

We follow [FSCIL](https://github.com/xyutao/fscil) setting to use the same data index_list for training.  
For CIFAR100, the dataset will be download automatically.  
For miniImagenet and CUB200, you can download
from [here](https://drive.google.com/drive/folders/11LxZCQj2FRCs0JTsf_dafvTHqFn2yGSN?usp=sharing). Please put the
downloaded file under `data/` folder and unzip it:

    $ tar -xvf miniimagenet.tar 
    $ tar -xvzf CUB_200_2011.tgz

Our pretrained parameters can be found in `pretrain`

## Training scripts

cifar100

    $ python main.py -dataset cifar100 -epochs_base 100 -lr_base 0.0001 -batch_size_base 512 -model_dir YOUR_MODEL_DIR -dataroot YOUR_DATA_PATH

mini_imagenet

    $ python main.py -dataset mini_imagenet -epochs_base 100 -lr_base 0.0001 -batch_size_base 256 -model_dir YOUR_MODEL_DIR -dataroot YOUR_DATA_PATH

cub200

    $ python main.py -dataset cub200 -epochs_base 100 -lr_base 0.0003 -batch_size_base 64 -model_dir YOUR_MODEL_DIR -dataroot YOUR_DATA_PATH

## Pretrain scripts

cifar100

    $python pretrain.py -dataset cifar100 -epochs 200 -batch_size 256 -dataroot YOUR_DATA_PATH

mini_imagenet

    $python pretrain.py -dataset mini_imagenet -epochs 200 -batch_size 512 -dataroot YOUR_DATA_PATH

cub200

    $python pretrain.py -dataset cub200 -epochs 200 -batch_size 256 -dataroot YOUR_DATA_PATH

## Acknowledgment

Our project references the codes in the following repos.

- [fscil](https://github.com/xyutao/fscil)
- [DeepEMD](https://github.com/icoz69/DeepEMD)
