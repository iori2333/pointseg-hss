import torch.nn as nn
import torch.nn.functional as F

class SeBranch(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 3)
        self.bn1 = nn.BatchNorm1d(32)
        self.maxPooling = nn.MaxPool1d(5, 5)
        self.conv2 = nn.Conv1d(32, 20, 5)
        self.bn2 = nn.BatchNorm1d(20)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        input = input.unsqueeze(1)
        x_padding = F.pad(input, [2, 2])
        x_conv1 = self.relu(self.bn1(self.conv1(x_padding)))
        x_unsample = self.maxPooling(x_conv1)
        x_unsample_padding = F.pad(x_unsample, [2, 2])
        feature = self.relu(self.bn2(self.conv2(x_unsample_padding)))
        return feature


class TwoCNN(nn.Module):

    def __init__(self, in_channels=500, channels=600, num_classes=19):
        super().__init__()
        self.seBranch = SeBranch()
        self.head = nn.Sequential(
            nn.Linear(in_channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 400),
            nn.ReLU(),
            nn.Linear(400, num_classes),
        )

    def forward(self, x):
        x = self.seBranch(x)
        x = x.view(x.shape[0], -1)
        x = self.head(x)
        return x
