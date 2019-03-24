import torch
from torch import nn
import torch.nn.functional as F


class RecurrentStepGenerationModel(nn.Module):

    def __init__(self):
        super(RecurrentStepGenerationModel, self).__init__()

        self.conv1a = nn.Conv2d(2, 10, (3, 7))
        self.conv2a = nn.Conv2d(10, 20, (3, 3))

        self.conv1b = nn.Conv2d(2, 10, (7, 3))
        self.conv2b = nn.Conv2d(10, 20, (3, 3))

        self.lstm = nn.LSTM(
                input_size=2 * 20 * 7 * 72 + 5 + 1,
                hidden_size=100,
                batch_first=True,
                num_layers=2)

        self.fc1 = nn.Linear(100, 128)

        # 4 softmax, 5 options each.
        self.fc2 = nn.Linear(128, 4 * 5)

    def forward(self, x, diff, placement_preds):
        # Placement preds should be the output of the step placement model.
        # Not jointly trained model.
        # Assumes that chunk_sizes match.
        batch_size, chunk_size, channels, timesteps, freqs = x.shape

        x = x.view((batch_size * chunk_size, channels, timesteps, freqs))

        xa = F.max_pool2d(F.relu(self.conv1a(x)), (3, 1), stride=1)
        xa = F.max_pool2d(F.relu(self.conv2a(xa)), (3, 1), stride=1)
        xa = xa.view((batch_size, self.chunk_size, 20 * 7 * 72))

        xb = F.max_pool2d(F.relu(self.conv1b(x)), (1, 3), stride=1)
        xb = F.max_pool2d(F.relu(self.conv2b(xb)), (1, 3), stride=1)
        xb = xb.view((batch_size, self.chunk_size, 20 * 7 * 72))

        x = torch.cat([xa, xb, diff, placement_preds], 2)

        x, hidden = self.lstm(x)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
        
class RegularizedRecurrentStepGenerationModel(nn.Module):

    def __init__(self):
        super(RegularizedRecurrentStepGenerationModel, self).__init__()

        self.conv1a = nn.Conv2d(2, 10, (3, 7))
        self.bn1a = nn.BatchNorm2d(10)
        self.conv2a = nn.Conv2d(10, 20, (3, 3))
        self.bn2a = nn.BatchNorm2d(20)

        self.conv1b = nn.Conv2d(2, 10, (7, 3))
        self.bn1b = nn.BatchNorm2d(10)
        self.conv2b = nn.Conv2d(10, 20, (3, 3))
        self.bn2b = nn.BatchNorm2d(20)

        self.lstm = nn.LSTM(
                input_size=2 * 20 * 7 * 72 + 5 + 1,
                hidden_size=100,
                batch_first=True,
                dropout=0.5,
                num_layers=2)

        self.fc1 = nn.Linear(100, 128)

        # 4 softmax, 5 options each.
        self.fc2 = nn.Linear(128, 4 * 5)

    def forward(self, x, diff, placement_preds):
        # Placement preds should be the output of the step placement model.
        # Not jointly trained model.
        # Assumes that chunk_sizes match.
        batch_size, chunk_size, channels, timesteps, freqs = x.shape

        x = x.view((batch_size * chunk_size, channels, timesteps, freqs))

        xa = F.relu(self.bn1a(self.conv1a(x)))
        xa = F.max_pool2d(xa, (3, 1), stride=1)
        xa = F.relu(self.bn2a(self.conv2a(xa)))
        xa = F.max_pool2d(xa, (3, 1), stride=1)
        xa = xa.view((batch_size, chunk_size, 20 * 7 * 72))

        xb = F.relu(self.bn1b(self.conv1b(x)))
        xb = F.max_pool2d(xb, (1, 3), stride=1)
        xb = F.relu(self.bn2b(self.conv2b(xb)))
        xb = F.max_pool2d(xb, (1, 3), stride=1)
        xb = xa.view((batch_size, chunk_size, 20 * 7 * 72))

        x = torch.cat([xa, xb, diff, placement_preds], 2)

        x, hidden = self.lstm(x)

        x = F.dropout(F.relu(self.fc1(x)), 0.5)
        x = self.fc2(x)

        return x
        
