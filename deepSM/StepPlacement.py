import torch
from torch import nn
import torch.nn.functional as F


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()

        self.linear = nn.Linear(3 * 15 * 80, 1)

    def forward(self, x):
        batch_size, channels, timesteps, freqs = x.shape
        x = x.view((batch_size, -1))
        x = self.linear(x)

        return torch.sigmoid(x)

class ConvStepPlacementModel(nn.Module):

    def __init__(self):
        super(ConvStepPlacementModel, self).__init__()

        self.conv1 = nn.Conv2d(2, 10, (3, 7))
        self.conv2 = nn.Conv2d(10, 20, (3, 3))

        self.fc1 = nn.Linear(20 * 7 * 72 + 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x, diff):
        batch_size, channels, timesteps, freqs = x.shape
        x = F.max_pool2d(F.relu(self.conv1(x)), (3, 1), stride=1)
        x = F.max_pool2d(F.relu(self.conv2(x)), (3, 1), stride=1)

        x = x.view((batch_size, 20 * 7 * 72))
        x = torch.cat([x, diff], 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x


class ConvStepPlacementModel(nn.Module):

    def __init__(self):
        super(ConvStepPlacementModel, self).__init__()

        self.conv1a = nn.Conv2d(2, 10, (3, 7))
        self.conv2a = nn.Conv2d(10, 20, (3, 3))

        self.conv1b = nn.Conv2d(2, 10, (7, 3))
        self.conv2b = nn.Conv2d(10, 20, (3, 3))

        self.fc1 = nn.Linear(2 * 20 * 7 * 72 + 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x, diff):
        batch_size, channels, timesteps, freqs = x.shape

        xa = F.max_pool2d(F.relu(self.conv1(x)), (3, 1), stride=1)
        xa = F.max_pool2d(F.relu(self.conv2(xa)), (3, 1), stride=1)
        xa = xa.view((batch_size, 20 * 7 * 72))

        xb = F.max_pool2d(F.relu(self.conv1(x)), (1, 3), stride=1)
        xb = F.max_pool2d(F.relu(self.conv2(xb)), (1, 3), stride=1)
        xb = xb.view((batch_size, 20 * 7 * 72))

        x = torch.cat([xa, xb, diff], 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x

class RecurrentStepPlacementModel(nn.Module):

    def __init__(self, chunk_size=200):
        super(RecurrentStepPlacementModel, self).__init__()

        self.chunk_size = chunk_size

        self.conv1a = nn.Conv2d(2, 10, (3, 7))
        self.conv2a = nn.Conv2d(10, 20, (3, 3))

        self.conv1b = nn.Conv2d(2, 10, (7, 3))
        self.conv2b = nn.Conv2d(10, 20, (3, 3))

        self.lstm = nn.LSTM(
                input_size=2 * 20 * 7 * 72 + 5,
                hidden_size=100,
                batch_first=True,
                num_layers=2)

        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, diff):
        # Assumes that chunk_sizes match.
        batch_size, chunk_size, channels, timesteps, freqs = x.shape

        x = x.view((batch_size * chunk_size, channels, timesteps, freqs))

        xa = F.max_pool2d(F.relu(self.conv1a(x)), (3, 1), stride=1)
        xa = F.max_pool2d(F.relu(self.conv2a(xa)), (3, 1), stride=1)
        xa = xa.view((batch_size, self.chunk_size, 20 * 7 * 72))

        xb = F.max_pool2d(F.relu(self.conv1b(x)), (1, 3), stride=1)
        xb = F.max_pool2d(F.relu(self.conv2b(xb)), (1, 3), stride=1)
        xb = xb.view((batch_size, self.chunk_size, 20 * 7 * 72))

        x = torch.cat([xa, xb, diff], 2)

        x, hidden = self.lstm(x)

        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x
        
