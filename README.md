# Incomplete-Multi-View-Clustering-via-Multi-Level-Contrastive-Learning
Code for "J Yin, P Wang, S Sun, Z Zheng. Incomplete Multi-View Clustering via Multi-Level Contrastive Learning, IEEE Transactions on Knowledge and Data Engineering, 2025"


## Requirements

pytorch==1.2.0 

numpy>=1.19.1

scikit-learn>=0.23.2

munkres>=1.1.4

## Datasets

The Caltech101-20, LandUse-21, CUB, Reuters, BDGP, NUSWIDE and Scene-15 datasets are placed in "data" folder. The NoisyMNIST dataset could be downloaded from [cloud](https://drive.google.com/file/d/1b__tkQMHRrYtcCNi_LxnVVTwB-TWdj93/view?usp=sharing).

## Configuration

The hyper-parameters, the training options (including **the missing rate and the learning rate**) are defined in configure.py.

## Usage

```bash
python run.py --dataset 0 --devices 0 --print_num 100 --test_time 5
```

## Files

run.py: This file is the main program of the code.
datasets.py: The purpose of this file is to processing data.
util.py: The purpose of this file is to initialize logs, normalize processing, and construct batch.
get_mask.py: The purpose of this file is to obtain three views of the incomplete data.
reloss.py: The purpose of this file is to calculate the loss.
model.py: The purpose of this file is to obtain our entire model.
configure.py: The purpose of this file is to set the parameters for each dataset.
metric.py: The purpose of this file is to obtain the ACC, ARI and AMI.
