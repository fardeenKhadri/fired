import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Conv2d (1 -> 16, kernel=3, padding=1)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # MaxPool (2x2)
        self.pool = nn.MaxPool2d(2, 2)
        # Conv2d (16 -> 32, kernel=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Linear (32*7*7 -> 128)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        # Linear (128 -> 10)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
