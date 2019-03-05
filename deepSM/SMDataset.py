import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler

from deepSM.smutils import SMFile
from deepSM import utils
import deepSM.beat_time_converter as BTC
from deepSM import wavutils
from deepSM import StepPlacement

import h5py

from importlib import reload
reload(BTC)
reload(wavutils)
reload(utils)

#
# TODO: MAKE CONSISTENT!
#


def get_dataset_from_file(dataset_name, song_names=None,
        chunk_size=200, base_path='datasets', n_songs=None, concat=True):

    ds_path = f"{base_path}/{dataset_name}"

    if song_names is None:
        song_names = os.listdir(ds_path)

    if n_songs is not None:
        song_names = song_names[:n_songs]

    smds = []
    for song_name in song_names:
        smds.append(load(song_name, dataset_name=dataset_name,
            chunk_size=chunk_size))

    if concat:
        return ConcatDataset(smds)
    else:
        return smds

def get_dataset_from_raw(song_names, base_path='.', chunk_size=200):
    smds = []
    for song_name in song_names:
        smds.append(generate(song_name, base_path, chunk_size=chunk_size))
    
    return ConcatDataset(smds)

def save_generated_datasets(song_names, dataset_name, base_path='datasets',
        data_path='.', test_split=0.25, overwrite=False):

    ds_path = f"{base_path}/{dataset_name}"

    if test_split is not None:
        song_names = song_names.copy()
        np.random.shuffle(song_names)

        n_train = int(np.round(len(song_names) * (1-test_split)))

        train_set = song_names[:n_train]
        test_set = song_names[n_train:]

        save_generated_datasets(
                train_set, dataset_name + '_train', base_path, data_path,
                None, overwrite)

        save_generated_datasets(
                test_set, dataset_name + '_test', base_path, data_path,
                None, overwrite)

        return

    if os.path.isdir(ds_path) and not overwrite:
        raise ValueError("Dataset already exists.")
    else:
        os.mkdir(ds_path)

    for song_name in song_names:
        smd = generate(song_name, data_path)
        smd.save(dataset_name=dataset_name)


def augment_dataset(dataset_name, model_name):
    new_dataset_name = f"{dataset_name}_placement"
    model_path = f"models/{model_name}"
    print("New dataset name:")
    print(new_dataset_name)

    device = torch.device('cuda:0')

    placement_model = StepPlacement.RecurrentStepPlacementModel(chunk_size=200)
    placement_model.load_state_dict(torch.load(model_path))
    placement_model.to(device)

    smds = get_dataset_from_file(dataset_name, chunk_size = -1, concat=False)

    for smd in smds:

        step_preds = []

        for i in range(len(smd.diffs)):
            d = smd[i]

            # Adding empty batch dimension.
            def preprocess_data(val):
                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val)
                return torch.unsqueeze(val, 0)

            d = dict(map(
                lambda x: (x[0], preprocess_data(x[1])),
                d.items()))


            step_pos_labels = d['step_pos_labels'].cuda()
            step_type_labels = d['step_type_labels'].cuda().long()
            fft_features = d['fft_features'].cuda()
            diff = d['diff'].cuda().float()


            with torch.no_grad():
                step_predictions = placement_model(fft_features, diff)
            

            step_predictions = np.r_[
                    np.zeros(smd.context_size),
                    step_predictions.cpu().numpy().reshape(-1),
                    np.zeros(smd.context_size)
            ].reshape((1, -1))

            step_preds.append(step_predictions)

        smd.step_predictions = np.concatenate(step_preds)
        smd.save(dataset_name=new_dataset_name)


def generate(song_name, base_path='.', chunk_size=200, context_size=7):
    """
    Generate an SMDataset from SM/wav files.
    Only creates datasets with no step predictions.
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
    # labels = {} # List of note aligned labels for note events. {0, 1} for now.


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

    step_pos_labels = np.zeros((len(diffs), N_frames))
    step_type_labels = np.zeros((len(diffs), N_frames, 4))
    for i, diff in enumerate(diffs):
        # Adjusting for the new frames added on to the front.
        frames[diff] += front_pad_frames

        # Generating final frame-aligned labels for note event:
        step_pos_labels[i, frames[diff]] = 1


        for j, note in zip(frames[diff], notes[diff]):
            step_type_labels[i, j, :] = np.array(list(map(int, note)))


    return SMDataset(song_name, fft_features, step_pos_labels, step_type_labels,
            diffs, chunk_size, context_size)

class SMDataset(Dataset):
    """
    Dataset loader for note placement network.
    Loads and feature engineers the songs and sm files for training.

    Note: Frame context size is currently hard coded!
    """

    def __init__(self, song_name, fft_features, step_pos_labels,
            step_type_labels, diffs, chunk_size, context_size,
            step_predictions=None):

        # Dataset properties.
        self.song_name = song_name
        self.fft_features = fft_features
        self.step_pos_labels = step_pos_labels
        self.step_type_labels = step_type_labels
        self.diffs = diffs

        # May be null.
        if step_predictions is not None:
            assert isinstance(step_predictions, np.ndarray)
        self.step_predictions = step_predictions

        self.N_frames = fft_features.shape[1]
        self.context_size = int(context_size)


        # Parse chunk size. 
        self.conv = False
        if chunk_size > 0: 
            self.chunk_size = int(chunk_size) 
        elif chunk_size == -1:
            # Get maximum chunk size, ie. sample_length == 1
            self.chunk_size = self.N_frames - self.context_size * 2
        elif:
            # Used for conv models.
            self.conv = True
            self.chunk_size = 1


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

        if self.chunk_size:
            fft_slice = slice(frame_idx, frame_idx + self.chunk_size + window_size-1)
            window_slice = slice(frame_idx + self.context_size,
                    frame_idx + self.context_size + self.chunk_size)

            feature_window = torch.from_numpy(self.fft_features[:,fft_slice,:])

            fft_features = feature_window.unfold(1, window_size, 1) 
            fft_features = fft_features.transpose(2, 3).transpose(0, 1)


            diff_mtx = np.zeros((self.chunk_size, 5))
            diff_mtx[:, diff_code] = 1

            step_pos_labels = self.step_pos_labels[diff_idx, window_slice]
            step_type_labels = self.step_type_labels[diff_idx, window_slice, :]

            if self.step_predictions is not None:

                step_predictions= \
                        self.step_predictions[diff_idx, window_slice]

                res = {
                    'fft_features': fft_features.float(),
                    'diff': diff_mtx.astype(np.float32),
                    'step_pos_labels': step_pos_labels.astype(np.float32),
                    'step_type_labels': step_type_labels.astype(np.float32),
                    'step_predictions': step_predictions.astype(np.float32)
                }
            else:
                res = {
                    'fft_features': fft_features.float(),
                    'diff': diff_mtx.astype(np.float32),
                    'step_pos_labels': step_pos_labels.astype(np.float32),
                    'step_type_labels': step_type_labels.astype(np.float32)
                }


        else:
            # Convolutional version.

            # First portion is unusable.
            frame_idx += self.context_size
            chunk_slice = slice(frame_idx - self.context_size,
                    frame_idx + self.context_size + 1)

            # Get the slice of the features/labels for the chunk.
            fft_features = self.fft_features[:,chunk_slice, :]

            step_pos_labels = np.array(self.step_pos_labels[diff_idx, frame_idx])
            step_type_labels = self.step_type_labels[diff_idx, frame_idx, :]

            
            diff_vec = np.zeros(5)
            diff_vec[diff_code] = 1

            res = {
                'fft_features': fft_features.astype(np.float32),
                'diff': diff_vec.astype(np.float32),
                'step_pos_labels': step_pos_labels.astype(np.float32),
                'step_type_labels': step_type_labels.astype(np.float32)
            }

        return res


    def save(self, dataset_name, base_path='datasets', fname=None):
        if fname is None:
            song_name = self.song_name
            fname = '%s/%s/%s/%s.h5' % \
                    (base_path, dataset_name, song_name, song_name)

        if not os.path.isdir('/'.join([base_path,dataset_name])):
            os.mkdir('/'.join([base_path,dataset_name]))
            
        if not os.path.isdir('/'.join([base_path,dataset_name,song_name])):
            os.mkdir('/'.join([base_path,dataset_name,song_name]))
                             

        with h5py.File(fname, 'w') as hf:

            hf.attrs['song_name'] = self.song_name
            hf.attrs['diffs'] = np.array(self.diffs, dtype='S10')
            hf.attrs['context_size'] = self.context_size

            hf.create_dataset('fft_features', data=self.fft_features)
            hf.create_dataset('step_pos_labels', data=self.step_pos_labels)
            hf.create_dataset('step_type_labels', data=self.step_type_labels)

            if self.step_predictions is not None:
                hf.create_dataset('step_predictions',
                        data=self.step_predictions)


def load(fname, dataset_name='base', chunk_size=200, base_path='datasets'):
    h5name = f'{base_path}/{dataset_name}/{fname}/{fname}.h5'
    with h5py.File(h5name, 'r') as hf:

        song_name = hf.attrs['song_name']
        diffs = list(map(lambda x: x.decode('ascii'), hf.attrs['diffs']))
        context_size = hf.attrs['context_size']

        fft_features = hf['fft_features'].value
        step_pos_labels = hf['step_pos_labels'].value
        step_type_labels = hf['step_type_labels'].value

        if 'step_predictions' in hf:
            step_predictions = hf['step_predictions'].value
        else:
            step_predictions = None

        return SMDataset(song_name, fft_features, step_pos_labels,
                step_type_labels, diffs, chunk_size, context_size,
                step_predictions)

