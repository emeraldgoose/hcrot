# hcrot
> This repository implements deep learning as python. 

## Requirements
```
pip install tqdm
```

## Example code
The mnist_test.csv file in /datasets is used as the dataset.
```shell
# MLP
python mlp.py --lr_rate 1e-2 --hidden_size 10 --epochs 10

# CNN
python cnn.py --lr_rate 1e-1 --epochs 10

# RNN
python rnn.py --model rnn --lr_rate 1e-3 --hidden_size 256 --epochs 10

# LSTM
python rnn.py --model lstm --lr_rate 1e-3 --hidden_state 256 --epochs 10
```

## Problem
Softmax and Sigmoid have a process of calculating in square form like e^i. If the input value suddenly becomes large during the learning process, overflow may occur.
> Error Message `Overflow (34, Numerical result out of range)`

## Post
| Layer | Link |
|-|-|
| MLP | [Link](https://emeraldgoose.github.io/pytorch/dl-implement/) |
| CNN | [Link](https://emeraldgoose.github.io/pytorch/cnn-implementation/) |
| RNN | [Link](https://emeraldgoose.github.io/pytorch/rnn-impl/) |
| LSTM | [Link](https://emeraldgoose.github.io/pytorch/lstm-implementation/)
