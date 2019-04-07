
import os
import shutil

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler

from deepSM import SMData
from deepSM import utils
import deepSM.beat_time_converter as BTC
from deepSM import wavutils
from deepSM import StepPlacement
from deepSM import SMDataset
from deepSM import SMGenDataset

import h5py

from importlib import reload
reload(BTC)
reload(wavutils)
reload(utils)
reload(SMData)
reload(SMDataset)


def get_dataset_from_file(
        dataset_name,
        dataset_type,
        song_names=None,
        n_songs=None,
        base_path=utils.BASE_PATH,
        concat=True,
        step_pos_labels=False,
        **kwargs):

    ds_path = f"{base_path}/datasets/{dataset_name}"
    dataset_type = dataset_type.lower()

    if song_names is None:
        song_names = os.listdir(ds_path)

    if n_songs is not None:
        song_names = song_names[:n_songs]

    smds = []

    # Can include context size here!
    for song_name in song_names:
        if dataset_type in 'generat':
            smds.append(SMGenDataset.load(
                song_name, dataset_name=dataset_name,
                **kwargs))
        elif dataset_type in 'placement':
            smds.append(SMDataset.load(
                song_name, dataset_name=dataset_name,
                **kwargs))
        else:
            raise ValueError("Dataset type must be gen or placement.")

    if concat:
        res =  ConcatDataset(smds)
    else:
        res =  smds

    if step_pos_labels:
        labels = []
        for smd in smds:
            labels.append(smd.step_pos_labels[:, 7:-7].reshape(-1))

        labels = np.concatenate(labels)

        return res, labels

    else:
        return res

def get_dataset_from_raw(
        raw_data_name,
        base_path=utils.BASE_PATH,
        chunk_size=100):

    song_names = os.listdir(f"{base_path}/data/{raw_data_name}")

    smds = []
    for song_name in song_names:
        smds.append(generate(
            song_name, raw_data_name, base_path, chunk_size=chunk_size))

    return ConcatDataset(smds)


def train_test_split_dataset(
        dataset_name,
        test_split=0.25,
        base_path=utils.BASE_PATH):

    ds_path = f"{base_path}/datasets/{dataset_name}"

    song_names = os.listdir(ds_path)
    np.random.shuffle(song_names)

    n_train = int(np.round(len(song_names) * (1 - test_split)))
    train_songs = song_names[:n_train]
    test_songs = song_names[n_train:]

    train_path = f"{ds_path}_train"
    test_path = f"{ds_path}_test"
    if not os.path.isdir(train_path):
        os.mkdir(train_path)
    if not os.path.isdir(test_path):
        os.mkdir(test_path)

    for song in train_songs:
        shutil.copytree(f"{ds_path}/{song}", f"{train_path}/{song}")

    for song in test_songs:
        shutil.copytree(f"{ds_path}/{song}", f"{test_path}/{song}")



def save_generated_datasets(
        raw_data_name,
        dataset_name=None,
        song_names=None,
        base_path=utils.BASE_PATH,
        test_split=0.25,
        overwrite=False,
        **kwargs):
    """
    Generates datasets from raw data, and saves them into a dataset.
    Optionally with train-test splits.
    """

    if dataset_name is None:
        dataset_name = f"{raw_data_name}_placement"

    raw_data_path = f"{base_path}/data/{raw_data_name}"
    ds_path = f"{base_path}/datasets/{dataset_name}"

    if song_names is None:
        song_names = os.listdir(raw_data_path)

    if test_split is not None:
        song_names = song_names.copy()
        np.random.shuffle(song_names)

        n_train = int(np.round(len(song_names) * (1-test_split)))

        train_set = song_names[:n_train]
        test_set = song_names[n_train:]

        save_generated_datasets(
                raw_data_name,
                dataset_name + '_train',
                train_set,
                base_path,
                None, overwrite, **kwargs)

        save_generated_datasets(
                raw_data_name,
                dataset_name + '_test',
                test_set,
                base_path,
                None, overwrite, **kwargs)

        return dataset_name

    if os.path.isdir(ds_path):
        if not overwrite:
            raise ValueError("Dataset %s already exists." % dataset_name)
    else:
        os.mkdir(ds_path)

    for song_name in song_names:
        smd = generate(song_name, raw_data_name, base_path, **kwargs)
        smd.save(dataset_name=dataset_name)

    return dataset_name


def augment_dataset(
        dataset_name, model_name, base_path=utils.BASE_PATH, **kwargs):
    """
    Adds step predictions to the dataset.

    Currently unused, as we train step generation models based on true step
    placement labels.
    """

    new_dataset_name = f"{dataset_name}_placement"
    model_path = f"{base_path}/models/{model_name}"
    print("New dataset name:")
    print(new_dataset_name)

    placement_model = StepPlacement.RegularizedRecurrentStepPlacementModel()
    placement_model.load_state_dict(torch.load(model_path))
    placement_model.cuda()

    smds = get_dataset_from_file(
            dataset_name, chunk_size = -1, concat=False, **kwargs)

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
                    np.ones(smd.context_size) * -25,
                    step_predictions.cpu().numpy().reshape(-1),
                    np.ones(smd.context_size) * -25
            ].reshape((1, -1))

            step_preds.append(step_predictions)

        smd.step_predictions = np.concatenate(step_preds)

        smd.save(dataset_name=new_dataset_name)


def generate(
        song_name,
        raw_data_name,
        base_path=utils.BASE_PATH,
        chunk_size=100,
        context_size=7,
        drop_diffs=[],
        log=False):
    """
    Generate an SMDataset from SM/wav files.
    Only creates datasets with no step predictions.
    """

    sm = SMData.SMFile(song_name, raw_data_name, base_path)

    # May want to save the time mapping later.
    btc = BTC.BeatTimeConverter(sm.offset, sm.bpms, sm.stops)

    # Will want to mantain order.
    # List of strings, not ints.
    diffs = list(filter(lambda x: x != 'Edit', sm.note_charts.keys()))
    if drop_diffs is not None:
        diffs = list(filter(lambda x: x not in drop_diffs, diffs))

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
    front_pad_frames, padded_wav = \
            wavutils.pad_wav(first_frame, last_frame, sm.wavdata)

    fft_features = wavutils.gen_fft_features(padded_wav, log=log)

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


    return SMDataset.SMDataset(
            song_name, diffs, fft_features, step_pos_labels, step_type_labels,
            chunk_size, context_size)
