import pandas as pd
import numpy as np
import argparse
from tqdm.auto import tqdm

from hcrot import layers, dataset, optim

def get_sinusoid_encoding_table(n_seq, d_hidn):
    # refs: https://paul-hyun.github.io/transformer-01/
    def cal_angle(position, i_hidn):
        return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)
    def get_posi_angle_vec(position):
        return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

    sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # even index sin 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # odd index cos

    return sinusoid_table

class TransformerClassifier(layers.Module):
    def __init__(self, embed_size=28, num_heads=7, hidden_dim=256, num_layers=2, num_classes=10, seq_length=28):
        super().__init__()
        self.embed_size = embed_size
        self.positional_encoding = np.expand_dims(get_sinusoid_encoding_table(seq_length, embed_size), axis=0)
        self.transformer = layers.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.flatten = layers.Flatten(start_dim=1)
        self.fc = layers.Linear(seq_length * embed_size, num_classes)

    def forward(self, src, tgt):
        src += self.positional_encoding[:, :src.shape[1], :]
        tgt += self.positional_encoding[:, :tgt.shape[1], :]
        
        output = self.transformer(src, tgt)
        
        out = self.flatten(output)
        out = self.fc(out)
        return out

def train(model, train_loader, criterion, optimizer):
    total_loss = 0
    for images, labels in tqdm(train_loader):
        tgt = np.zeros_like(images)
        outputs = model.forward(images, tgt)
        loss = criterion(outputs, labels)
        dz = criterion.backward()
        optimizer.update(dz)
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 평가 함수
def evaluate(model, test_loader, criterion):
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in tqdm(test_loader):
        tgt = np.zeros_like(images)
        outputs = model.forward(images, tgt)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        predicted = outputs.argmax(1)
        total += labels.shape[0]
        correct += (predicted==labels).sum().item()

    accuracy = 100 * correct / total
    return total_loss / len(test_loader), accuracy

def main(args):
    model = TransformerClassifier(hidden_dim=args.hidden_size)
    criterion = layers.CrossEntropyLoss()
    optimizer = optim.Adam(model, args.lr_rate)
    
    loss_, accuracy = [], []
    for epoch in range(args.epochs):
        train_loss = train(model, dataloader, criterion, optimizer)
        test_loss, test_accuracy = evaluate(model, testloader, criterion)
        loss_.append(train_loss)
        accuracy.append(test_accuracy)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hcrot example training code')
    parser.add_argument('--lr_rate', default=1e-3, type=float, help='Learning Rate')
    parser.add_argument('--epochs', default=5, type=int, help='Epochs')
    parser.add_argument('--hidden_size', default=256, type=int, help='Transformer hidden size')

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
    main(args)
    