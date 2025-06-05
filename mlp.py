import pandas as pd
import numpy as np
import argparse
from tqdm.auto import tqdm

from hcrot import layers, dataset, optim

class Model(layers.Module):
    def __init__(self, input_len=28*28, hidden=512, num_classes=10):
        super().__init__()
        self.linear1 = layers.Linear(in_features=input_len, out_features=hidden)
        self.sigmoid1 = layers.Sigmoid()
        self.dropout = layers.Dropout(p=0.3)
        self.linear2 = layers.Linear(in_features=hidden, out_features=hidden)
        self.sigmoid2 = layers.Sigmoid()
        self.fc = layers.Linear(in_features=hidden, out_features=num_classes)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.sigmoid2(x)
        o = self.fc(x)
        return o

def train(args):
    model = Model(input_len=28*28, hidden=args.hidden_size, num_classes=10)
    criterion = layers.CrossEntropyLoss()
    optimizer = optim.Adam(model, args.lr_rate)

    for epoch in range(args.epochs):
        loss_, correct = 0, 0
        
        # train
        model.train()
        for x, y in tqdm(dataloader):
            pred = model(x)
            loss = criterion(pred,y)
            dz = criterion.backward()
            optimizer.update(dz)
            loss_ += loss
        
        # test
        model.eval()
        for x, y in tqdm(testloader):
            pred = model(x)
            correct += np.sum(np.argmax(pred,axis=1)==y)

        print(f'epoch = [{epoch+1}] | loss = {loss_/len(dataloader)} | ACC = {correct/(len(testloader)*len(y))}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
