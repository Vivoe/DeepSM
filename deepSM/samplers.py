import torch
from torch.utils.data import Sampler

import numpy as np


class WeightedRandomSampler(Sampler):
    def __init__(self, labels, size=None, pos_prob=0.5):
        # Assumes binary labels.

        N = len(labels)
        if size is None:
            size = N
        self.size = size

        pos_weight = pos_prob / np.mean(labels)
        neg_weight = (1 - pos_prob) / (1 - np.mean(labels))
        
        probs = np.where(labels == 1, pos_weight, neg_weight)
        probs /= np.sum(probs)

        self.sequence = np.random.choice(np.arange(N), size=self.size,
                p=probs)

    def __iter__(self):
        return iter(self.sequence)

    def __len__(self):
        # Really, should be infinite, not sure what len is really used for.
        return self.size

class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """

        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except:
            print('Need scikit-learn for this functionality')
        import numpy as np
        
        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = torch.randn(self.class_vector.size(0),2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)
