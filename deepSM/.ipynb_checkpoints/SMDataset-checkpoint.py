import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler

from deepSM.smutils import SMFile
from deepSM import utils
import deepSM.beat_time_converter as BTC
from deepSM import wavutils

import h5py

from importlib import reload
reload(BTC)
reload(wavutils)
reload(utils)


def get_dataset_from_file(song_names, base_path='.', data_folder='data'):
    smds = []
    for song_name in song_names:
        smds.append(load(song_name, base_path, data_folder))

    return ConcatDataset(smds)

def get_dataset_from_raw(song_names, base_path='.'):
    smds = []
    for song_name in song_names:
        smds.append(generate(song_name, base_path))
    
    return ConcatDataset(smds)

def save_generated_datasets(song_names, base_path='.'):
    for song_name in song_names:
        smd = generate(song_name, base_path)
        smd.save()


def generate(song_name, base_path='.', chunk_size=100, context_size=7):
    """
    Generate an SMDataset from SM/wav files.
    """

    sm = SMFile(song_name, base_path)

    # May want to save the time mapping later.
    btc = BTC.BeatTimeConverter(sm.offset, sm.bpms, sm.stops)

    # Will want to mantain order.
    # List of strings, not ints.
    diffs = list(filter(lambda x: x != 'Edit', sm.note_charts.keys()))

    notes = {} # Contains only a list of notes for each difficulty.
    times = {} # List of times per diff.
    frames = {}
#    labels = {} # List of note aligned labels for note events. {0, 1} for now.


    # Track first and last notes for wav padding.
    first_frame = np.inf
    last_frame = -np.inf

    # Find note times and frames for alignment to features.
    for diff in diffs:
        times[diff], notes[diff] = \
            btc.gen_time_notes(sm.note_charts[diff].notes)

        frames[diff] = btc.align_to_frame(times[diff])

        if frames[diff][0] < first_frame:
            first_frame = frames[diff][0]
        if frames[diff][-1] > last_frame:
            last_frame = frames[diff][-1]

    # Test this!
    # Test by writing beeps again.
    front_pad_frames, padded_wav = wavutils.pad_wav(first_frame, last_frame, sm.wavdata)

    fft_features = wavutils.gen_fft_features(padded_wav)

    # N_channels = 3 (1024, 2048, 4096)
    # N_frames ~ song length * 44100 / 512
    # N_freqs = 80 (Number of mel coefs per frame)
    N_channels, N_frames, N_freqs = fft_features.shape

    # Number of possible starting frames in the song.
    # Need to exclude ending lag and unusable frames at the very ends.
    sample_length = N_frames - chunk_size - context_size * 2


    labels = np.zeros((len(diffs), N_frames))
    for i, diff in enumerate(diffs):
        # Adjusting for the new frames added on to the front.
        frames[diff] += front_pad_frames

        # Generating final frame-aligned labels for note event:
        labels[i, frames[diff]] = 1

    # Testing alignment of frames.
    # wavutils.test_alignment(padded_wav, frames[diff] * 512 / 44100)

    return SMDataset(song_name, fft_features, labels, diffs, chunk_size,
            context_size)

class SMDataset(Dataset):
    """
    Dataset loader for note placement network.
    Loads and feature engineers the songs and sm files for training.

    Note: Frame context size is currently hard coded!
    """

    def __init__(self, song_name, fft_features, labels, diffs, chunk_size,
            context_size):

        # Dataset properties.
        self.song_name = song_name
        self.fft_features = fft_features
        self.labels = labels
        self.diffs = diffs
        self.chunk_size = chunk_size
        self.context_size = context_size

        # Genratable from dataset properties.
        self.N_frames = fft_features.shape[1]
        self.sample_length = self.N_frames - self.chunk_size - self.context_size * 2
        


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

        # First self.context_size frames are unusable.
        frame_idx += self.context_size

        diff = self.diffs[diff_idx]
        diff_code = utils.difficulties[diff]

        # chuck_slice = slice(frame_idx, frame_idx + self.chunk_size)
        chunk_slice = slice(frame_idx - self.context_size, frame_idx + self.context_size + 1)

        # Get the slice of the features/labels for the chunk.
        fft_features = self.fft_features[:,chunk_slice, :]

        # event_labels = self.labels[diff][chunk_slice]
        # CNN version. currently scalar.
        event_labels = self.labels[diff_idx, frame_idx].reshape((1))
        
        diff_vec = np.zeros(5)
        diff_vec[diff_code] = 1

        res = {
            'fft_features': fft_features.astype(np.float32),
            'diff': diff_vec.astype(np.float32),
            'labels': event_labels.astype(np.float32)
        }

        return res


    def save(self, base_path='.', data_folder='data', fname=None):
        if fname is None:
            song_name = self.song_name
            fname = '%s/%s/%s/%s.h5' % (base_path, data_folder, song_name, song_name)
        
        print(fname)
        if not os.path.isdir('/'.join([base_path,data_folder])):
            os.mkdir('/'.join([base_path,data_folder]))
            
        if not os.path.isdir('/'.join([base_path,data_folder,song_name])):
            os.mkdir('/'.join([base_path,data_folder,song_name]))
                             

        with h5py.File(fname, 'w') as hf:

            hf.attrs['song_name'] = self.song_name
            hf.attrs['diffs'] = np.array(self.diffs, dtype='S10')
            hf.attrs['chunk_size'] = self.chunk_size
            hf.attrs['context_size'] = self.context_size

            hf.create_dataset('fft_features', data=self.fft_features)
            hf.create_dataset('labels', data=self.labels)


def load(fname, base_path='.', data_folder='data'):
    h5name = base_path + f'/{data_folder}/{fname}/{fname}.h5'
    with h5py.File(h5name, 'r') as hf:

        song_name = hf.attrs['song_name']
        diffs = list(map(lambda x: x.decode('ascii'), hf.attrs['diffs']))
        chunk_size = hf.attrs['chunk_size']
        context_size = hf.attrs['context_size']

        fft_features = hf['fft_features'].value
        labels = hf['labels'].value

        return SMDataset(song_name, fft_features, labels, diffs, chunk_size,
            context_size)
