import os
import h5py
import numpy as np

from deepSM import utils

import torch
import torch.utils.data as datautils

from importlib import reload

__version = 'Gen-1-0'

def load(song_name, dataset_name, base_path=utils.BASE_PATH, **kwargs):
    fname = f'{base_path}/datasets/{dataset_name}/{song_name}/{song_name}.h5'

    with h5py.File(fname, 'r') as hf:
        assert song_name == hf.attrs['song_name']
        # Convert back from fixed length strings.
        diff_names = hf.attrs['diff_names'].astype(str)

        diff_data = {}
        for diff_name in diff_names:
            hf_data = hf[diff_name]
            d = {}

            # Should include:
            # fft_features
            # beats_before
            # beats_after
            # step_type_labels
            # Note: Cannot share fft_features between diffs due to different
            # indices.
            for key in hf_data.keys():
                d[key] = hf_data[key].value

            diff_data[diff_name] = d

    return SMGenDataset(
            song_name,
            diff_names,
            diff_data,
            **kwargs)




class SMGenDataset(datautils.Dataset):

    def __init__(
            self,
            song_name,
            diff_names,
            diff_data,
            chunk_size=10,
            base_path=utils.BASE_PATH):
        """
        diff_data = {
            "diff_name": {
                "fft_features": ...,
                "beats_before": ...,
                "beats_after": ...,
                "step_type_labels?": ...
            }
        }
        """

        self.song_name = song_name
        self.diff_names = diff_names
        self.diff_data = diff_data
        self.chunk_size = chunk_size

        k = list(diff_data.keys())[0]
        self.use_labels = 'step_type_labels' in diff_data[k]

        # Calculate at which index each difficulty will start.
        # Get length of each difficulty.
        for diff in self.diff_data:
            d = self.diff_data[diff]
            if chunk_size == -1:
                self.diff_data[diff]['diff_len'] = 1
            else:
                self.diff_data[diff]['diff_len'] = \
                        d['fft_features'].shape[0] - chunk_size + 1

        # glorified cumsum
        self.diff_start_idx = [0]
        for i, diff_name in enumerate(self.diff_names):
            self.diff_start_idx.append(
                    self.diff_start_idx[-1] + \
                    self.diff_data[diff_name]['diff_len'])


    def __len__(self):
        # Cannot simplify chunk size, since lengths between difficulties vary.
        if self.chunk_size == -1:
            return len(self.diff_data)

        l = 0
        for key in self.diff_data.keys():
            l += self.diff_data[key]['fft_features'].shape[0] \
                    - self.chunk_size + 1

        return l


    def __getitem__(self, idx):
        diff_idx = np.digitize(idx, self.diff_start_idx) - 1
        assert diff_idx >= 0 and diff_idx < len(self.diff_start_idx)
        frame_idx = idx - self.diff_start_idx[diff_idx]

        diff_name = self.diff_names[diff_idx]
        diff_code = utils.difficulties[diff_name]

        d = self.diff_data[diff_name]

        # Take everything.
        if self.chunk_size == -1:
            window_slice = slice(None, None)
            diff = np.zeros((d['fft_features'].shape[0], 5))
        else:
            window_slice = slice(frame_idx, frame_idx + self.chunk_size)
            diff = np.zeros((self.chunk_size, 5))


        fft_features = d['fft_features'][window_slice]
        beats_before = d['beats_before'][window_slice]
        beats_after = d['beats_after'][window_slice]
        diff[:, diff_code] = 1
        if self.use_labels:
            step_type_labels = d['step_type_labels'][window_slice]

        res = {
            'fft_features': fft_features,
            'beats_before': beats_before,
            'beats_after': beats_after,
            'diff': diff
        }

        if self.use_labels:
            res['step_type_labels'] = step_type_labels

        return res
