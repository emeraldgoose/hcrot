# hcrot
> 딥러닝을 Python으로 구현하는 레포지토리 입니다.

- 최대한 Pytorch와 비슷하게 구현하려고 했습니다. Tensor 대신에 Numpy 클래스를 사용합니다.

## Requirements
```
pip install tqdm
```

## Example code
/datasets 안에 mnist_test.csv 파일을 데이터셋으로 합니다.  
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

## Error
Softmax와 Sigmoid는 e^i처럼 제곱형태로 계산하는 과정이 있습니다. 학습 과정 중에 입력되는 값이 갑자기 커지게 되면 오버플로우가 발생할 수 있습니다.  
> `Overflow (34, Numerical result out of range)`

## Post
| Layer | Link |
|-|-|
| MLP | [Link](https://emeraldgoose.github.io/pytorch/dl-implement/) |
| CNN | [Link](https://emeraldgoose.github.io/pytorch/cnn-implementation/) |
| RNN | [Link](https://emeraldgoose.github.io/pytorch/rnn-impl/) |
| LSTM | [Link](https://emeraldgoose.github.io/pytorch/lstm-implementation/)
