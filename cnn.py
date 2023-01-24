import pandas as pd
import numpy as np
import argparse
from tqdm.auto import tqdm

from hcrot import layers, dataset, optim

class CNN(layers.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.layer1 = layers.Sequential(layers.Conv2d(1,5,5), layers.ReLU(), layers.MaxPool2d(2,2))
        self.layer2 = layers.Sequential(layers.Conv2d(5,7,5), layers.ReLU(), layers.MaxPool2d(2,2))
        self.flatten = layers.Flatten()
        self.dropout2 = layers.Dropout(p=0.5)
        self.fc = layers.Linear(112, num_classes)

    def forward(self, x):
        for module in self.sequential:
            x = module(x)
        return x

def train(args):
    model = CNN()
    criterion = layers.CrossEntropyLoss()
    optimizer = optim.Adam(model,args.lr_rate)
    for epoch in range(args.epochs):
        loss_, correct = 0, 0
        
        # train
        model.train()
        for i, (x,y) in enumerate(tqdm(dataloader)):
            x = np.array(x).reshape(-1,1,28,28) # (B,H,W,C) -> (B,C,H,W)
            pred = model.forward(x)
            loss = criterion(pred,y)
            dz = criterion.backward()
            optimizer.update(dz)
            loss_ += loss
        
        # test
        model.eval()
        for i, (x,y) in enumerate(tqdm(testloader)):
            x = np.array(x).reshape(-1,1,28,28)
            pred = model.forward(x)
            correct += np.sum(np.argmax(pred,axis=1)==y)
        
        print(f'epoch = [{epoch+1}] | loss = {loss_/len(dataloader)} | ACC = {correct/(len(testloader)*len(y))}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hcrot example training code')
    parser.add_argument('--lr_rate', default=1e-2, type=float, help='Learning Rate')
    parser.add_argument('--epochs', default=10, type=int, help='Epochs')

    df = pd.read_csv('./datasets/mnist_test.csv')
    label = df['7'].to_numpy()
    df = df.drop('7',axis=1)
    dat = df.to_numpy()

    train_image, test_image = dat[:5000], dat[8001:9001]
    train_label, test_label = label[:5000], label[8001:9001]
    train_image = train_image.astype(np.float32)
    test_image = test_image.astype(np.float32)

    train_image_, test_image_ = [], []
    for i in range(len(train_image)): train_image_.append(train_image[i].reshape(28,28) / 255.)
    for i in range(len(test_image)): test_image_.append(test_image[i].reshape(28,28) / 255.)

    dataloader = dataset.Dataloader(train_image_, train_label, batch_size=50, shuffle=True)
    testloader = dataset.Dataloader(test_image_, test_label, batch_size=10, shuffle=False)

    args = parser.parse_args()
    train(args)