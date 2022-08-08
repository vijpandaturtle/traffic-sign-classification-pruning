# DL Library imports
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        #Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        #Dense layers
        self.fc1 = nn.Linear(256*7*7, 1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=256)
        self.fc3 = nn.Linear(256, output_dim)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(F.max_pool2d(self.conv5(x), kernel_size=2))
        #Flatten
        h = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(h))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x