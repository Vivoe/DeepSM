import os
import h5py
import numpy as np

from deepSM import utils

import torch
import torch.utils.data as datautils

from importlib import reload

__version = 'Gen-1-0'

class SMGenDataset(datautils.Dataset):

    def __init__(
            self,
            song_name,
            dataset_name,
            chunk_size=10,
            base_path=utils.BASE_PATH):

        fname = f'{base_path}/datasets/{dataset_name}/{song_name}/{song_name}.h5'

        self.song_name = song_name
        self.chunk_size = chunk_size

        with h5py.File(fname, 'r') as hf:
            assert self.song_name == hf.attrs['song_name']
            # Convert back from fixed length strings.
            self.diff_names = hf.attrs['diff_names'].astype(str)

            self.diff_data = {}
            for diff_name in self.diff_names:
                d = hf[diff_name]
                self.diff_data[diff_name] = {}

                # Should include:
                # fft_features
                # beats_before
                # beats_after
                # step_type_labels
                # Note: Cannot share fft_features between diffs due to different
                # indices.
                for key in d.keys():
                    self.diff_data[diff_name][key] = d[key].value

                self.diff_data[diff_name]['diff_len'] = \
                        self.diff_data[diff_name]['fft_features'].shape[0]\
                        - self.chunk_size + 1

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
            l += diff_data[key]['fft_features'].shape[0] - self.chunk_size + 1

        return l


    def __getitem__(self, idx):
        diff_idx = np.digitize(idx, self.diff_start_idx) - 1
        assert diff_idx >= 0 and diff_idx < len(self.diff_start_idx)
        frame_idx = idx - self.diff_start_idx[diff_idx]

        diff_name = self.diff_names[diff_idx]
        diff_code = utils.difficulties[diff_name]

        d = self.diff_data[diff_name]

        window_slice = slice(frame_idx, frame_idx + self.chunk_size)

        fft_features = d['fft_features'][window_slice]
        beats_before = d['beats_before'][window_slice]
        beats_after = d['beats_after'][window_slice]
        step_type_labels = d['step_type_labels'][window_slice]
        diff = np.zeros((self.chunk_size, 5))
        diff[:, diff_code] = 1

        return {
            'fft_features': fft_features,
            'beats_before': beats_before,
            'beats_after': beats_after,
            'diff': diff,
            'step_type_labels': step_type_labels
        }
