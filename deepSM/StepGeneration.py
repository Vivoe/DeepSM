import torch
from torch import nn
import torch.nn.functional as F

from deepSM import NNModel
from deepSM import utils


class RegularizedRecurrentStepGenerationModel(NNModel.NNModel):

    def __init__(self, log=True):
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
                # Conv size + diff + beats_before + beats_after
                input_size= 20 * 2 * 35 + 20 * 7 * 8 + 5 + 1 + 1,
                hidden_size=100,
                batch_first=True, dropout=0.5,
                num_layers=2)

        self.fc1 = nn.Linear(100, 128)

        # 4 softmax, 5 options each.
        self.fc2 = nn.Linear(128, 4 * 5)

    def forward(self, x, diff, beats_before, beats_after):
        # Placement preds should be the output of the step placement model.
        # Not jointly trained model.
        # Assumes that chunk_sizes match.
        batch_size, chunk_size, channels, timesteps, freqs = x.shape

        x = x.view((batch_size * chunk_size, channels, timesteps, freqs))

        xa = F.relu(self.bn1a(self.conv1a(x)))
        xa = F.max_pool2d(xa, (3, 1), stride=(2, 2))
        xa = F.relu(self.bn2a(self.conv2a(xa)))
        xa = F.max_pool2d(xa, (3, 1), stride=1)
        xa = xa.view((batch_size, chunk_size, 20 * 2 * 35))

        xb = F.relu(self.bn1b(self.conv1b(x)))
        xb = F.max_pool2d(xb, (1, 3), stride=(1, 3))
        xb = F.relu(self.bn2b(self.conv2b(xb)))
        xb = F.max_pool2d(xb, (1, 3), stride=(1, 3))
        xb = xb.view((batch_size, chunk_size, 20 * 7 * 8))

        x = torch.cat([xa, xb, diff, beats_before, beats_after], 2)

        x, hidden = self.lstm(x)

        x = F.dropout(F.relu(self.fc1(x)), 0.5)
        x = self.fc2(x)

        return x

    def get_criterion(self):
        weights = torch.tensor([1, 1, 2, 2, 2]).float()
        if self.use_cuda:
            weights = weights.cuda()
        return nn.CrossEntropyLoss(weight=weights)

    def compute_loss(self, criterion, outputs, step_type_labels):
        # loss = torch.zeros(1).cuda()
        loss = 0
        for j in range(4):
            loss += criterion(outputs[:,:,j*5:(j+1)*5].view((-1, 5)),
                              step_type_labels[:,:,j].view(-1))

        return loss

    def prepare_data(self, batch, use_labels=True):
        if use_labels:
            labels = batch['step_type_labels'].long()
        fft_features = batch['fft_features'].float()
        batch_size = fft_features.shape[0]
        beats_before = batch['beats_before'].unsqueeze(2).float()
        beats_after = batch['beats_after'].unsqueeze(2).float()
        diff = batch['diff'].float()

        if self.use_cuda:
            if use_labels:
                labels = labels.cuda()
            fft_features = fft_features.cuda()
            beats_before = beats_before.cuda()
            beats_after = beats_after.cuda()
            diff = diff.cuda()

        if use_labels:
            return (fft_features, diff, beats_before, beats_after), labels
        else:
            return fft_features, diff, beats_before, beats_after

