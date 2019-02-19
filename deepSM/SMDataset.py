import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler

from deepSM.smutils import SMFile
from deepSM import utils
import deepSM.beat_time_converter as BTC
from deepSM import wavutils

from importlib import reload
reload(BTC)
reload(wavutils)
reload(utils)


def get_dataset(song_names):
    smds = []
    for song_name in song_names:
        smds.append(SMDataset(song_name))

    return ConcatDataset(smds)


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


class SMDataset(Dataset):
    """
    Dataset loader for note placement network.
    Loads and feature engineers the songs and sm files for training.

    Note: Frame context size is currently hard coded!
    """

    def __init__(self, song_name, chunk_length=100):
        self.song_name = song_name
        self.chunk_length = chunk_length

        self.sm = SMFile(song_name)
        btc = BTC.BeatTimeConverter(self.sm.offset, self.sm.bpms, self.sm.stops)

        # Will want to mantain order.
        self.available_diffs = list(filter(lambda x: x != 'Edit', self.sm.note_charts.keys()))

        self.notes = {} # Contains only a list of notes for each difficulty.
        self.times = {} # List of times per diff.
        self.frames = {}
        self.labels = {} # List of note aligned labels for note events. {0, 1} for now.


        # Track first and last notes for wav padding.
        first_frame = np.inf
        last_frame = -np.inf

        # Find note times and frames for alignment to features.
        for diff in self.available_diffs:
            self.times[diff], self.notes[diff] = \
                btc.gen_time_notes(self.sm.note_charts[diff].notes)

            self.frames[diff] = btc.align_to_frame(self.times[diff])

            if self.frames[diff][0] < first_frame:
                first_frame = self.frames[diff][0]
            if self.frames[diff][-1] > last_frame:
                last_frame = self.frames[diff][-1]

        # Test this!
        # Test by writing beeps again.
        front_pad_frames, padded_wav = wavutils.pad_wav(first_frame, last_frame, self.sm.wavdata)



        self.fft_features = wavutils.gen_fft_features(padded_wav)

        # N_channels = 3 (1024, 2048, 4096)
        # N_frames ~ song length * 44100 / 512
        # N_freqs = 80 (Number of mel coefs per frame)
        self.N_channels, self.N_frames, self.N_freqs = self.fft_features.shape

        # Number of possible starting frames in the song.
        # Need to exclude ending lag and unusable frames at the very ends.
        self.sample_length = self.N_frames - self.chunk_length - 7 * 2


        for diff in self.available_diffs:
            # Adjusting for the new frames added on to the front.
            self.frames[diff] += front_pad_frames

            # Generating final frame-aligned labels for note event:
            self.labels[diff] = np.zeros(self.N_frames)
            self.labels[diff][self.frames[diff]] = 1

        # Testing alignment of frames.
        # wavutils.test_alignment(padded_wav, self.frames[diff] * 512 / 44100)


    def __len__(self):
        # Can start at any point in the song, as long as there is enough
        # room to unroll to chunk_length.
        return len(self.available_diffs) * (self.N_frames - self.chunk_length - 7 * 2)


    def __getitem__(self, idx):
        # Since all difficulties have the same number of frames, divide to get
        # which diff, order determined by self.available_diffs.
        # Remainder to find the frame.
        # "Concatenated" representation.
        diff_idx = idx // self.sample_length
        frame_idx = idx % self.sample_length

        # First 7 frames are unusable.
        frame_idx += 7

        diff = self.available_diffs[diff_idx]
        diff_code = utils.difficulties[diff]

        # chuck_slice = slice(frame_idx, frame_idx + self.chunk_length)
        chunk_slice = slice(frame_idx - 7, frame_idx + 7 + 1)

        # Get the slice of the features/labels for the chunk.
        fft_features = self.fft_features[:,chunk_slice, :]

        # event_labels = self.labels[diff][chunk_slice]
        event_labels = self.labels[diff][frame_idx].reshape((1))

        res = {
            'fft_features': fft_features.astype(np.float32),
            'diff': diff_code,
            'labels': event_labels.astype(np.float32)
        }

        return res