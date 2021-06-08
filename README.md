# Controllable Gradient Item Retrieval

This repository is the PyTorch implementation of Controllable Gradient Item Retrieval (WWW 21).

[arXiv](https://arxiv.org/abs/2106.00062)

If you make use of the code/experiment, please cite our paper (Bibtex below).

```
@inproceedings{wang2021controllable,
  title={Controllable Gradient Item Retrieval},
  author={Wang, Haonan and Zhou, Chang and Yang, Carl and Yang, Hongxia and He, Jingrui},
  booktitle={Proceedings of the Web Conference 2021},
  pages={768--777},
  year={2021}
}
```


Contact: Haonan Wang (haonan3@illinois.edu)


## Installation
Install PyTorch following the instuctions on the [official website] (https://pytorch.org/). The code has been tested over PyTorch 1.1.0 versions.

Then install the other dependencies.

```
conda env create -f environment.yml
```


## Test run

Please download ml-20m and ml-25m from [movielens](https://grouplens.org/datasets/movielens/), and then put them to `data/ml-20m/raw/` and `data/ml-25m/raw/` respectively.

Hyper-parameters need to be specified through the commandline arguments.

For item retrieval experiment, please refer `run.sh`.