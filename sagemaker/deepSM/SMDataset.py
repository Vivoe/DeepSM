import os
import io

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler

from deepSM.SMData import SMFile
from deepSM import utils
import deepSM.beat_time_converter as BTC
from deepSM import wavutils

import h5py

from importlib import reload
reload(BTC)
reload(wavutils)
reload(utils)

__version__ = '2-0-0'

def load(f, **kwargs):
    if isinstance(f, bytes):
        f = io.BytesIO(f)

    with h5py.File(f, 'r') as hf:

        song_name = hf.attrs['song_name']
        assert song_name == song_name

        diffs = hf.attrs['diffs'].astype(str)
        # diffs = list(map(lambda x: x.decode('ascii'), hf.attrs['diffs']))

        fft_features = hf['fft_features'].value
        step_pos_labels = hf['step_pos_labels'].value
        step_type_labels = hf['step_type_labels'].value

    return SMDataset(
            song_name,
            diffs,
            fft_features,
            step_pos_labels,
            step_type_labels,
            **kwargs)

class SMDataset(Dataset):
    """
    Dataset loader for note placement network.
    Loads and feature engineers the songs and sm files for training and prediction.

    For chunk size:
        Use None for Conv models.
        Use -1 for prediction (Returns entire song as a single sequence.)
        Use N for RNNs of sequence length N.

    Will ignore all labels if either set of labels is None.
    """

    def __init__(
            self,
            song_name,
            diffs,
            fft_features,
            step_pos_labels=None,
            step_type_labels=None,
            chunk_size=100,
            context_size=7):

        self.song_name = song_name
        self.diffs = diffs
        self.fft_features = fft_features
        self.step_pos_labels = step_pos_labels
        self.step_type_labels = step_type_labels

        if step_pos_labels is None or step_type_labels is None:
            self.use_labels = False
        else:
            self.use_labels = True

        self.N_frames = self.fft_features.shape[1]
        self.context_size = int(context_size)


        # Parse chunk size.
        self.conv = False
        if chunk_size is None:
            # Used for conv models.
            self.conv = True
            self.chunk_size = 1

        elif chunk_size > 0:
            self.chunk_size = int(chunk_size)

        elif chunk_size == -1:
            # Get maximum chunk size, ie. sample_length == 1
            self.chunk_size = self.N_frames - self.context_size * 2
        else:
            raise ValueError("Invalid chunk size.")


        # Genratable from dataset properties.
        self.sample_length = self.N_frames \
                - self.chunk_size - self.context_size * 2 + 1



    def __len__(self):
        # Can start at any point in the song, as long as there is enough
        # room to unroll to chunk_size.
        return len(self.diffs) * self.sample_length


    def __getitem__(self, idx):
        # Since all difficulties have the same number of frames, divide to get
        # which diff, order determined by self.diffs.
        # Remainder to find the frame.
        # "Concatenated" representation.
        diff_idx = idx // self.sample_length
        frame_idx = idx % self.sample_length


        diff = self.diffs[diff_idx]
        diff_code = utils.difficulties[diff]

        window_size = self.context_size * 2 + 1

        fft_slice = slice(frame_idx, frame_idx + self.chunk_size + window_size-1)
        window_slice = slice(frame_idx + self.context_size,
                frame_idx + self.context_size + self.chunk_size)

        feature_window = torch.from_numpy(self.fft_features[:,fft_slice,:])

        fft_features = feature_window.unfold(1, window_size, 1)
        fft_features = fft_features.transpose(2, 3).transpose(0, 1)


        diff_mtx = np.zeros((self.chunk_size, 5))
        diff_mtx[:, diff_code] = 1

        # Get appropriate labels.
        if self.use_labels:
            step_pos_labels = self.step_pos_labels[diff_idx, window_slice]
            step_type_labels = self.step_type_labels[diff_idx, window_slice, :]

        # Aggregate data into final dictionary.
        if self.conv:
            res = {
                'fft_features': torch.squeeze(fft_features.float(), 0),
                'diff': diff_mtx.astype(np.float32).reshape(-1),
            }
        else:
            res = {
                'fft_features': fft_features.float(),
                'diff': diff_mtx.astype(np.float32),
            }

        if self.use_labels:
            res['step_pos_labels'] = step_pos_labels.astype(np.float32)
            res['step_type_labels'] = step_type_labels.astype(np.float32)

        return res


    def save(self, buf):
        # Takes in a bytes-like object and writes to that.
#         with h5py.File(buf) as hf:
        with h5py.File(buf) as hf:
            hf.attrs['song_name'] = self.song_name
            hf.attrs['diffs'] = np.array(self.diffs, dtype='S10')
            hf.attrs['context_size'] = self.context_size

            hf.create_dataset('fft_features', data=self.fft_features)
            hf.create_dataset('step_pos_labels', data=self.step_pos_labels)
            hf.create_dataset('step_type_labels', data=self.step_type_labels)
        
        return buf