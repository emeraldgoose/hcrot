> 딥러닝을 최대한 Python 기본 라이브러리로만 구현하는 레포지토리 입니다.

- 최대한 Pytorch와 비슷하게 구현하려고 했습니다. 그래서 패키지명을 torch의 반대방향인 hcrot로 지었습니다.
- pandas는 데이터를 불러올때만 사용했습니다.
- numpy는 dot product 구현에 사용했습니다. dot product를 3중 for문으로 구현하게 되면 너무 오래걸려 MNIST 1000개 학습에 6분 이상 걸리게 됩니다.
- Cross Entropy는 계산을 쉽게 하기 위해 numpy로 구현했지만 나중에 numpy 없이 구현할 생각입니다.