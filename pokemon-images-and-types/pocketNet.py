import torch.nn as nn
import torch.nn.functional as F

class pocketNet(nn.Module):
    def __init__(self):
        super(pocketNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 13)
        self.conv2 = nn.Conv2d(16, 32, 13)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.conv4 = nn.Conv2d(64, 128, 5)
        self.pooling1 = nn.MaxPool2d(7, 2)
        self.pooling2 = nn.MaxPool2d(7, 2)
        self.fc1 = nn.Linear(128 * 14 * 14, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 19)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pooling1(x)
        x = F.relu(self.conv3(x))
        x = self.pooling2(x)
        x = F.relu(self.conv4(x))
        x = x.view(-1, 128 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
