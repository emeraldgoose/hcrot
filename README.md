> 딥러닝을 최대한 Python 기본 라이브러리로만 구현하는 레포지토리 입니다.

- 최대한 Pytorch와 비슷하게 구현하려고 했습니다. 그래서 패키지명을 torch의 반대방향인 hcrot로 지었습니다.
- pandas는 데이터를 불러올때만 사용했습니다.
- numpy는 dot product 구현에 사용했습니다. dot product를 3중 for문으로 구현하게 되면 너무 오래걸려 MNIST 1000개 학습에 6분 이상 걸리게 됩니다.
- Cross Entropy는 계산을 쉽게 하기 위해 numpy로 구현했지만 나중에 numpy 없이 구현할 생각입니다.

## Example code
/datasets 안에 mnist.csv 파일을 학습하는 예졔 코드입니다.  
`python main.py --lr_rate 1e-2 --hidden_size 10 --epochs 10`

## 현재 문제점
Softmax와 Sigmoid는 e^i처럼 제곱형태로 계산하는 과정이 있습니다. 학습 과정 중에 입력되는 값이 갑자기 커지게 되면 오버플로우가 발생할 수 있습니다.  
- `Overflow (34, Numerical result out of range)`
- round로 입력되는 값과 출력하는 값을 반올림하여 임시로 해결했지만 가끔씩 발생하고 있습니다.

## Post
[https://emeraldgoose.github.io/pytorch/dl-implement/](https://emeraldgoose.github.io/pytorch/dl-implement/)
