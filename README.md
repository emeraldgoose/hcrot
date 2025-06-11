# hcrot
> This repository implements deep learning as python. 

## Requirements
```
pip install numpy pandas tqdm
```

## Problem
Softmax and Sigmoid have a process of calculating in square form like e^i. If the input value suddenly becomes large during the learning process, overflow may occur.
> Error Message `Overflow (34, Numerical result out of range)`

## Post
| Layer | Link | notebook |
|-|-|-|
| MLP | [Link](https://emeraldgoose.github.io/pytorch/dl-implement/) | [notebook](./notebooks/mlp.ipynb) |
| CNN | [Link](https://emeraldgoose.github.io/pytorch/cnn-implementation/) | [notebook](./notebooks/cnn.ipynb) |
| RNN | [Link](https://emeraldgoose.github.io/pytorch/rnn-impl/) | [notebook](./notebooks/rnn.ipynb) |
| LSTM | [Link](https://emeraldgoose.github.io/pytorch/lstm-implementation/) | [notebook](./notebooks/rnn.ipynb) |
| Transformer | [Link1](https://emeraldgoose.github.io/pytorch/transformer-scratch-implementation-1/), [Link2](https://emeraldgoose.github.io/pytorch/transformer-scratch-implementation-2/) | [notebook](./notebooks/transformer.ipynb), [simple_LM](./notebooks/simple_LM.ipynb), [simple_gpt](./notebooks/simple_gpt.ipynb) |
| Diffusion | [Link1](https://emeraldgoose.github.io/pytorch/text-to-image-implementation/), [Link2](https://emeraldgoose.github.io/pytorch/unet-and-ddpm-implementation/) | [notebook](./notebooks/diffusion.ipynb) |