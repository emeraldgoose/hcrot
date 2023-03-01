import pandas as pd
import numpy as np
import argparse
from tqdm.auto import tqdm

from hcrot import layers, dataset, optim

class RNN(layers.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.rnn = layers.RNN(
            input_size=28,
            hidden_size=hidden_size,
            num_layers=2,
            nonlinearity='tanh',
            batch_first=False
            )
        self.fc = layers.Linear(hidden_size, 10)
    
    def forward(self, x):
        x,_ = self.rnn(x)
        return self.fc(x[-1])

def train(args):
    model = RNN(args.hidden_size)
    criterion = layers.CrossEntropyLoss()
    optimizer = optim.Adam(model,args.lr_rate)

    for epoch in range(args.epochs):
        loss_, correct = 0, 0
        for image, label in tqdm(dataloader):
            image = np.transpose(image, (1,0,2)) # (Length, Batch, Features)
            pred = model.forward(image)
            loss = criterion(pred, label)
            dz = criterion.backward()
            optimizer.update(dz)
            loss_ += loss.item()

        for image, label in tqdm(testloader):
            image = np.transpose(image, (1,0,2))
            pred = model.forward(image)
            correct += np.sum(np.argmax(pred,axis=1)==label)

        print(f'{epoch+1} / {args.epochs} | loss = {loss_/len(dataloader)} | ACC = {correct/(len(testloader)*len(label))}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hcrot example training code')
    parser.add_argument('--lr_rate', default=1e-3, type=float, help='Learning Rate')
    parser.add_argument('--epochs', default=10, type=int, help='Epochs')
    parser.add_argument('--hidden_size', default=256, type=int, help='RNN hidden size')

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