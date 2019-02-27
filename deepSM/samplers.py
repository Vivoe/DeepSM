import torch
from torch.utils.data import Sampler

import numpy as np


class WeightedRandomSampler(Sampler):
    def __init__(self, dataset):

        N = len(dataset)
        labels = np.zeros(N)
        for i in range(N):
            labels[i] = dataset[i]['labels'][0]

        N_steps = np.sum(labels)
        print("Original probability:", np.mean(labels))
        sample_probs = np.where(labels, (N - N_steps) / N_steps, 1)

        self.weights = torch.from_numpy(sample_probs)

    def __iter__(self):
        return iter(torch.multinomial(self.weights, len(self.weights)))


    def __len__(self):
        # Really, should be infinite, not sure what len is really used for.
        return len(self.weights)

