import torch
from torch import nn
import torch.nn.functional as F

from . import NNModel

from importlib import reload
reload(NNModel)

print("Reloaded")



class ConvStepPlacementModel(nn.Module):

    def __init__(self):
        super(ConvStepPlacementModel, self).__init__()

        self.conv1 = nn.Conv2d(2, 10, (3, 7))
        self.conv2 = nn.Conv2d(10, 20, (3, 3))

        self.fc1 = nn.Linear(20 * 7 * 72 + 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x, diff):
        batch_size, channels, freqs = x.shape
        x = F.max_pool2d(F.relu(self.conv1(x)), (3, 1), stride=1)
        x = F.max_pool2d(F.relu(self.conv2(x)), (3, 1), stride=1)

        x = x.view((batch_size, 20 * 7 * 72))
        x = torch.cat([x, diff], 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x


class DualConvStepPlacementModel(nn.Module):

    def __init__(self):
        super(DualConvStepPlacementModel, self).__init__()

        self.conv1a = nn.Conv2d(2, 10, (3, 7))
        self.conv2a = nn.Conv2d(10, 20, (3, 3))

        self.conv1b = nn.Conv2d(2, 10, (7, 3))
        self.conv2b = nn.Conv2d(10, 20, (3, 3))

        self.fc1 = nn.Linear(2 * 20 * 7 * 72 + 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x, diff):
        batch_size, channels, timesteps, freqs = x.shape

        xa = F.max_pool2d(F.relu(self.conv1a(x)), (3, 1), stride=1)
        xa = F.max_pool2d(F.relu(self.conv2a(xa)), (3, 1), stride=1)
        xa = xa.view((batch_size, 20 * 7 * 72))

        xb = F.max_pool2d(F.relu(self.conv1b(x)), (1, 3), stride=1)
        xb = F.max_pool2d(F.relu(self.conv2b(xb)), (1, 3), stride=1)
        xb = xb.view((batch_size, 20 * 7 * 72))

        x = torch.cat([xa, xb, diff], 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x


class DropoutConvStepPlacementModel(nn.Module):

    def __init__(self):
        super(DropoutConvStepPlacementModel, self).__init__()

        self.conv1a = nn.Conv2d(2, 10, (3, 7))
        self.conv2a = nn.Conv2d(10, 20, (3, 3))

        self.conv1b = nn.Conv2d(2, 10, (7, 3))
        self.conv2b = nn.Conv2d(10, 20, (3, 3))

        self.fc1 = nn.Linear(2 * 20 * 7 * 72 + 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x, diff):
        batch_size, channels, timesteps, freqs = x.shape

        xa = F.max_pool2d(F.relu(self.conv1a(x)), (3, 1), stride=1)
        xa = F.max_pool2d(F.relu(self.conv2a(xa)), (3, 1), stride=1)
        xa = xa.view((batch_size, 20 * 7 * 72))

        xb = F.max_pool2d(F.relu(self.conv1b(x)), (1, 3), stride=1)
        xb = F.max_pool2d(F.relu(self.conv2b(xb)), (1, 3), stride=1)
        xb = xb.view((batch_size, 20 * 7 * 72))

        x = torch.cat([xa, xb, diff], 1)

        x = F.dropout(F.relu(self.fc1(x)), 0.5)
        x = F.dropout(F.relu(self.fc2(x)), 0.5)
        x = torch.sigmoid(self.fc3(x))

        return x


class RegularizedConvStepPlacementModel(NNModel.PlacementModel):
    def __init__(self):
        super().__init__()

        self.conv1a = nn.Conv2d(2, 10, (3, 7))
        self.bn1a = nn.BatchNorm2d(10)
        self.conv2a = nn.Conv2d(10, 20, (3, 3))
        self.bn2a = nn.BatchNorm2d(20)

        self.conv1b = nn.Conv2d(2, 10, (7, 3))
        self.bn1b = nn.BatchNorm2d(10)
        self.conv2b = nn.Conv2d(10, 20, (3, 3))
        self.bn2b = nn.BatchNorm2d(20)

        self.fc1 = nn.Linear(2 * 20 * 7 * 72 + 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x, diff):

        batch_size, channels, timesteps, freqs = x.shape

        # Conv path a
        xa = F.relu(self.bn1a(self.conv1a(x)))
        xa = F.max_pool2d(xa, (3, 1), stride=1)
        xa = F.relu(self.bn2a(self.conv2a(xa)))
        xa = F.max_pool2d(xa, (3, 1), stride=1)
        xa = xa.view((batch_size, 20 * 7 * 72))

        # Conv path b
        xb = F.relu(self.bn1b(self.conv1b(x)))
        xb = F.max_pool2d(xb, (1, 3), stride=1)

        xb = F.relu(self.bn2b(self.conv2b(xb)))
        xb = F.max_pool2d(xb, (1, 3), stride=1)
        xb = xa.view((batch_size, 20 * 7 * 72))


        # Merge into FC land.
        x = torch.cat([xa, xb, diff], 1)

        x = F.dropout(F.relu(self.fc1(x)), 0.5)
        x = F.dropout(F.relu(self.fc2(x)), 0.5)
        x = self.fc3(x)
        # x = torch.sigmoid(self.fc3(x))

        return x

    def compute_loss(self, criterion, outputs, labels):
        return criterion(torch.sigmoid(outputs), labels)

    def get_criterion(self):
        return nn.BCELoss()

    def prepare_data(self, batch):
        labels = batch['step_pos_labels']
        fft_features = batch['fft_features']
        diff = batch['diff']

        if self.use_cuda:
            labels = labels.cuda()
            fft_features = torch.log(fft_features).cuda()
            diff = diff.cuda()

        return (fft_features, diff), labels


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
        xa = xa.view((batch_size, chunk_size, 20 * 7 * 72))

        xb = F.max_pool2d(F.relu(self.conv1b(x)), (1, 3), stride=1)
        xb = F.max_pool2d(F.relu(self.conv2b(xb)), (1, 3), stride=1)
        xb = xb.view((batch_size, chunk_size, 20 * 7 * 72))

        x = torch.cat([xa, xb, diff], 2)
        x, hidden = self.lstm(x)

        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x


class RegularizedRecurrentStepPlacementModel(NNModel.PlacementModel):

    def __init__(self):
        super().__init__()

        self.conv1a = nn.Conv2d(2, 10, (3, 7))
        self.bn1a = nn.BatchNorm2d(10)
        self.conv2a = nn.Conv2d(10, 20, (3, 3))
        self.bn2a = nn.BatchNorm2d(20)

        self.conv1b = nn.Conv2d(2, 10, (7, 3))
        self.bn1b = nn.BatchNorm2d(10)
        self.conv2b = nn.Conv2d(10, 20, (3, 3))
        self.bn2b = nn.BatchNorm2d(20)

        self.lstm = nn.LSTM(
                input_size=2 * 20 * 7 * 72 + 5,
                hidden_size=100,
                batch_first=True,
                dropout=0.5,
                num_layers=2)

        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, diff):
        # Assumes that chunk_sizes match.
        batch_size, chunk_size, channels, timesteps, freqs = x.shape

        x = x.view((batch_size * chunk_size, channels, timesteps, freqs))

        # Conv path a
        xa = F.relu(self.bn1a(self.conv1a(x)))
        xa = F.max_pool2d(xa, (3, 1), stride=1)
        xa = F.relu(self.bn2a(self.conv2a(xa)))
        xa = F.max_pool2d(xa, (3, 1), stride=1)
        xa = xa.view((batch_size, chunk_size, 20 * 7 * 72))

        # Conv path b
        xb = F.relu(self.bn1b(self.conv1b(x)))
        xb = F.max_pool2d(xb, (1, 3), stride=1)

        xb = F.relu(self.bn2b(self.conv2b(xb)))
        xb = F.max_pool2d(xb, (1, 3), stride=1)
        xb = xa.view((batch_size, chunk_size, 20 * 7 * 72))

        x = torch.cat([xa, xb, diff], 2)

        x, hidden = self.lstm(x)

        x = F.dropout(F.relu(self.fc1(x)), 0.5)
        x = self.fc2(x)

        return x

    def compute_loss(self, criterion, outputs, labels):
        outputs = torch.squeeze(outputs, 2)
        return criterion(torch.sigmoid(outputs), labels)

    def get_criterion(self):
        return nn.BCELoss()

    def prepare_data(self, batch):
        labels = batch['step_pos_labels']
        fft_features = batch['fft_features']
        diff = batch['diff']

        if self.use_cuda:
            labels = labels.cuda()
            fft_features = fft_features.cuda()
            diff = diff.cuda()

        return (fft_features, diff), labels
