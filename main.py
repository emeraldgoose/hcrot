import pandas as pd
import numpy as np
import argparse
from tqdm.auto import tqdm

from hcrot import layers, dataset, optim, utils

class Model(object):
    def __init__(self, input_len=28*28, hidden=512, num_classes=10):
        super().__init__()
        self.linear = layers.Linear(in_features=input_len, out_features=hidden)
        self.linear2 = layers.Linear(in_features=hidden, out_features=hidden)
        self.fc = layers.Linear(in_features=hidden, out_features=num_classes)
        self.sigmoid = layers.Sigmoid()
        self.relu = layers.ReLU()
        self.sequential = [self.linear, self.sigmoid, self.linear2, self.sigmoid, self.fc]
        
    def forward(self, x):
        for module in self.sequential:
            x = module(x)
        return x

def train(args):
    model = Model(input_len=28*28,hidden=args.hidden_size,num_classes=10)
    loss_fn = layers.CrossEntropyLoss()
    optimizer = optim.Adam(model,args.lr_rate)

    for epoch in range(args.epochs):
        loss_, correct = 0, 0
        
        # train
        for i,(x,y) in enumerate(tqdm(dataloader)):
            pred = model.forward(x)
            loss = loss_fn(pred,y)
            dz = loss_fn.backward(pred, y)
            optimizer.update(dz)
            loss_ += loss
        
        # test
        for i, (x,y) in enumerate(tqdm(testloader)):
            pred = model.forward(x)
            correct += sum([(a==b) for a,b in zip(utils.argmax(pred),y)]) 

        print(f'epoch = [{epoch+1}] | loss = {loss_/len(dataloader)} | ACC = {correct/(len(testloader)*len(y))}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hcrot example training code')
    parser.add_argument('--lr_rate', default=1e-2, type=float, help='Learning Rate')
    parser.add_argument('--hidden_size', default=28, type=int, help='Hidden Layer size')
    parser.add_argument('--epochs', default=10, type=int, help='Epochs')

    df = pd.read_csv('./datasets/mnist_test.csv')
    label = df['7'].to_numpy()
    df = df.drop('7',axis=1)
    dat = df.to_numpy()

    train_image, test_image = dat[:5000], dat[8001:9001]
    train_label, test_label = label[:5000], label[8001:9001]
    train_image = train_image.astype(np.float32)
    test_image = test_image.astype(np.float32)

    for i in range(len(train_image)): train_image[i] /= 255.0
    for i in range(len(test_image)): test_image[i] /= 255.0

    dataloader = dataset.Dataloader(train_image, train_label, batch_size=50, shuffle=True)
    testloader = dataset.Dataloader(test_image, test_label, batch_size=10, shuffle=False)

    args = parser.parse_args()
    train(args)
