import torch
from torch import nn
import torch.nn.functional as F

from deepSM import NNModel


class RegularizedConvStepPlacementModel(NNModel.NNModel):
    def __init__(self, log=True):
        super().__init__()

        self.log = log

        # Sizes are time x freq


        # High freq. Would help resolve double notes but..
        self.conv1a = nn.Conv2d(2, 10, (3, 7))
        self.bn1a = nn.BatchNorm2d(10)
        self.conv2a = nn.Conv2d(10, 20, (3, 3))
        self.bn2a = nn.BatchNorm2d(20)

        # Long time.
        self.conv1b = nn.Conv2d(2, 10, (7, 3))
        self.bn1b = nn.BatchNorm2d(10)
        self.conv2b = nn.Conv2d(10, 20, (3, 3))
        self.bn2b = nn.BatchNorm2d(20)

        # self.fc1 = nn.Linear(2 * 20 * 7 * 72 + 5, 256)
        self.fc1 = nn.Linear(20 * 3 * 72 + 20 * 33 * 24 + 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x, diff):

        batch_size, channels, timesteps, freqs = x.shape

        # Conv path a
        xa = F.relu(self.bn1a(self.conv1a(x)))
        xa = F.max_pool2d(xa, (3, 1), stride=(3, 1))
        xa = F.relu(self.bn2a(self.conv2a(xa)))
        xa = F.max_pool2d(xa, (3, 1), stride=(3, 1))
        xa = xa.view((batch_size, 20 * 3 * 72))

        # Conv path b
        xb = F.relu(self.bn1b(self.conv1b(x)))
        xb = F.max_pool2d(xb, (1, 3), stride=1)

        xb = F.relu(self.bn2b(self.conv2b(xb)))
        xb = F.max_pool2d(xb, (1, 3), stride=(1, 3))
        xb = xb.view((batch_size, 20 * 33 * 24))


        # Merge into FC land.
        x = torch.cat([xa, xb, diff], 1)

        x = F.dropout(F.relu(self.fc1(x)), 0.5)
        x = F.dropout(F.relu(self.fc2(x)), 0.5)
        x = self.fc3(x)

        return x

    def compute_loss(self, criterion, outputs, labels):
        return criterion(outputs, labels)

    def get_criterion(self):
        return nn.BCEWithLogitsLoss()

    # def compute_loss(self, criterion, outputs, labels):
        # class_outputs = F.pad(outputs, (1, 0))
        # return criterion(class_outputs, labels[:,0])

    # def get_criterion(self):
        # weights = torch.tensor([1, 4]).float()
        # if self.use_cuda:
            # weights = weights.cuda()
        # return nn.CrossEntropyLoss(weight=weights)

    def prepare_data(self, batch):
        # labels = batch['step_pos_labels'].long()
        labels = batch['step_pos_labels'].float()
        fft_features = batch['fft_features']
        diff = batch['diff']

        if self.log:
            fft_features = torch.log10(fft_features)

        if self.use_cuda:
            labels = labels.cuda()
            fft_features = fft_features.cuda()
            diff = diff.cuda()

        return (fft_features, diff), labels


class RegularizedRecurrentStepPlacementModel(NNModel.NNModel):

    def __init__(self, log=True):
        super().__init__()

        self.log = log

        self.conv1a = nn.Conv2d(2, 10, (3, 7))
        self.bn1a = nn.BatchNorm2d(10)
        self.conv2a = nn.Conv2d(10, 20, (3, 3))
        self.bn2a = nn.BatchNorm2d(20)

        self.conv1b = nn.Conv2d(2, 10, (7, 3))
        self.bn1b = nn.BatchNorm2d(10)
        self.conv2b = nn.Conv2d(10, 20, (3, 3))
        self.bn2b = nn.BatchNorm2d(20)

        self.lstm = nn.LSTM(
                input_size=20 * 2 * 34 + 20 * 7 * 8 + 5,
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
        xa = F.max_pool2d(xa, (3, 2), stride=(2, 2))
        xa = F.relu(self.bn2a(self.conv2a(xa)))
        xa = F.max_pool2d(xa, (3, 2), stride=1)
        xa = xa.view((batch_size, chunk_size, 20 * 2 * 34))

        # Conv path b
        xb = F.relu(self.bn1b(self.conv1b(x)))
        xb = F.max_pool2d(xb, (1, 3), stride=(1, 3))

        xb = F.relu(self.bn2b(self.conv2b(xb)))
        xb = F.max_pool2d(xb, (1, 3), stride=(1, 3))
        xb = xb.view((batch_size, chunk_size, 20 * 7 * 8))

        x = torch.cat([xa, xb, diff], 2)

        x, hidden = self.lstm(x)

        x = F.dropout(F.relu(self.fc1(x)), 0.5)
        x = self.fc2(x)

        return x

    def compute_loss(self, criterion, outputs, labels):
        outputs = torch.squeeze(outputs, 2)
        # return criterion(torch.sigmoid(outputs), labels)
        return criterion(outputs, labels)

    def get_criterion(self):
        # return nn.BCELoss()
        return nn.BCEWithLogitsLoss()

    def prepare_data(self, batch, use_labels=True):
        if use_labels:
            labels = batch['step_pos_labels'].float()

        fft_features = batch['fft_features'].float()
        diff = batch['diff'].float()

        if self.log:
            fft_features = torch.log10(fft_features + 1e-6)

        if self.use_cuda:
            if use_labels:
                labels = labels.cuda()
            fft_features = fft_features.cuda()
            diff = diff.cuda()

        if use_labels:
            return (fft_features, diff), labels
        else:
            return fft_features, diff
