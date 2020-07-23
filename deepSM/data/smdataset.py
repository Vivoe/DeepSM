import numpy as np
import torch
import torch.utils.data as tdata

from concurrent.futures import ProcessPoolExecutor
import os
from tqdm import tqdm

from deepSM.data.beattimeconverter import BeatTimeConverter
from deepSM.data import smfile
from deepSM.data import audioprocessing as ap
from deepSM import utils

from importlib import reload
reload(smfile)

audio_config = utils.config['audio']



def getSMDataset(ds_path):
    song_paths = os.listdir(ds_path)

    smds = []
    for song_path in tqdm(song_paths):
        smds.append(SMDataset(f"{ds_path}/{song_path}"))

    return tdata.ConcatDataset(smds)


class SMDataset(tdata.Dataset):
    """
    Datset representing a single song.
    Loads data from SMFiles and audio files and their preprocessing.

    Each data point is a slice of a spectrogram for a given difficulty.
    """

    def __init__(self, fpath):
        """
        Currently does not do lazy loading.
        """
        super().__init__()


        # Index of array specifies difficulty.
        self.diffs, self.mel_data, self.note_labels = self.load(fpath)

        self.n_frames = self.mel_data.shape[-1]
        self.window_size = audio_config['contextWindowSize'] * 2 + 1
        self.n_windows = self.n_frames - self.window_size + 1


    def __len__(self):
        return len(self.diffs) * self.n_windows


    def __getitem__(self, idx):
        frame_idx = idx % self.n_windows + audio_config['contextWindowSize']
        diff_idx = idx // self.n_windows

        start_frame = frame_idx - audio_config['contextWindowSize']
        end_frame = frame_idx + audio_config['contextWindowSize'] + 1

        timing_labels = (
            self.note_labels[diff_idx, frame_idx, :]
                .sum().clamp_max(1)
        )

        fuzzy_timing_labels = (
            self.note_labels[diff_idx, frame_idx-1:frame_idx+1, :]
                .sum().clamp_max(1)
        )
        
        res = {
            'fft_features': self.mel_data[:, :, start_frame:end_frame],
            'diff': self.diffs[diff_idx],
            'dir_labels': self.note_labels[diff_idx, frame_idx, :],
            'timing_labels': timing_labels,
            'fuzzy_timing_labels': fuzzy_timing_labels
        }

        return res


    def load(self, fpath):
        sm_file = smfile.SMFile(fpath)

        audio_data = ap.load_raw_audio(fpath + '/' + sm_file.music)
        mel_data = ap.fft_processing(audio_data)

        btc = BeatTimeConverter(sm_file)

        # Get time, note, and frame data for each Chart in the SMFile
        diffs = []
        times = {}
        notes = {}
        frames = {}
        for diff, chart in sm_file.note_charts.items():
            times[diff], notes[diff] = btc.gen_time_notes(chart.notes)
            frames[diff] = btc.align_to_frame(times[diff])
            diffs.append(int(chart.diff_value))


        # Pad using frame data
        first_frame = min(map(lambda f: f[0], frames.values()))
        last_frame = min(map(lambda f: f[-1], frames.values()))

        mel_data, front_pad = ap.pad_audio(mel_data, first_frame, last_frame)

        n_frames = mel_data.shape[-1]

        # Use padding and frame data to creat note labels for each chart.
        note_labels = []
        for diff in sm_file.note_charts:

            note_label = torch.zeros(n_frames, 4)
            for frame, note in zip(frames[diff], notes[diff]):
                # Adjust note labels and frames to padded mel_data.
                note_label[frame + front_pad, :] = torch.ShortTensor(list(map(int, note)))
            
            note_labels.append(note_label)

        note_labels = torch.stack(note_labels)

        return (diffs, mel_data, note_labels)