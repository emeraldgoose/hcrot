import random

class Dataloader:
    def __init__(self, X, y, batch_size=1, shuffle=True):
        assert len(y) % batch_size == 0, "length must batch multiplcation"
        self.idx = [i for i in range(len(X))]
        self.data, self.label = X, y
        self.batch_size = batch_size
        if shuffle: random.shuffle(self.idx)

    def __len__(self):
        return len(self.idx) // self.batch_size

    def __getitem__(self, i):
        # numpy to python list
        images = [self.data[i*self.batch_size+j].tolist() for j in range(self.batch_size)]
        labels = [self.label[i*self.batch_size+j].tolist() for j in range(self.batch_size)]
        return images, labels