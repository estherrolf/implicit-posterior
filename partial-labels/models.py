import torch as T

class SmallCNN(T.nn.Module):
    def __init__(self, n_ch):
        super(SmallCNN, self).__init__()
        self.conv1 = T.nn.Conv2d(n_ch, 32, 3, 1, 1)
        self.conv2 = T.nn.Conv2d(32, 32, 3, 2, 1)
        self.conv3 = T.nn.Conv2d(32, 32, 3, 2, 1)
        self.conv4 = T.nn.Conv2d(32, 32, 3, 2, 1)
        self.fc = T.nn.Linear(32*4*4, 10)
    def forward(self, x):
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.conv3(x).relu()
        x = self.conv4(x).relu().view(-1,32*4*4)
        x = self.fc(x)
        return x