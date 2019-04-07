"""
Convert Datasets for the Step Placement models into datasets for the
Step Generation model.
"""

import os
import numpy as np
from sklearn.metrics import f1_score
import h5py
import warnings

from deepSM import SMData
from deepSM import SMDUtils
from deepSM import StepPlacement
from deepSM import utils
from deepSM import bpm_estimator

from torch.utils import data as datautils

__version__ = '1-0'

from importlib import reload
reload(utils)


def compute_thresh(model, dataset):
    """
    Not in use. See bin/get_thresholds.py.
    """

    # Currently unused, due to gen dataset directly using labels.
    # Should use this for final prediction pipeline.
    # Chosen from optimization of F1 score.
    loader = datautils.DataLoader(dataset)

    output_list, labels_list = model.predict(dataset, return_list=True)
    outputs = torch.cat(list(map(lambda x: x[0,:,0], output_list)))
    labels = torch.cat(list(map(lambda x: x[0,:], labels_list)))

    probs = torch.sigmoid(outputs).numpy()

    def f1_fn(thresh):
        preds = (probs > thresh).astype(int)
        return f1_score(labels, preds)

    scores = []
    threshes = range(1e-2, 0.5, 1000)
    for i in threshes:
        scores.append(f1_fn(i))

    thresh_idx = np.argmax(scores)
    thresh = threshes[thresh_idx]
    f1 = scores[thresh_idx]

    print(f"Threshold: {thresh} F1 score: {f1}")

    return thresh

def get_generation_features(smd, bpm, frame_idxs_list=None, use_labels=True):
    """
    Builds the step generation features from a step placement dataset.
    Computes the appropriate STFT features, changes in time, and the
    frame index of each timestep.

    frame_idxs_list should be a list of preds.
    If frame_idxs is None, then use step pos labels.
    """
    diff_order = []
    diff_features = {}

    # Iterate through each data point (difficulty) of the dataset.
    for i, d in enumerate(smd):
        # Mantain order of difficulties.
        diff_code = np.where(d['diff'][0])[0][0]
        diff = utils.inv_difficulties[diff_code] # String
        diff_order.append(diff)

        if frame_idxs_list is None:
            frame_idxs = np.where(d['step_pos_labels'])[0]
        else:
            frame_idxs = frame_idxs_list[i]


        fft_features = d['fft_features'][frame_idxs]

        if use_labels:
            step_type_labels = d['step_type_labels'][frame_idxs]

        # Compute delta_time, delta_beat
        bps = 60 / bpm
        delta_frame = frame_idxs[1:] - frame_idxs[:-1]
        delta_time = delta_frame * 512/44100
        delta_beat = delta_time / bps

        # 4 measures of time before and after the first and last notes.
        beats_before = np.r_[12, delta_beat]
        beats_after = np.r_[delta_beat, 12]

        diff_features[diff] = {
                'fft_features': fft_features,
                'beats_before': beats_before,
                'beats_after': beats_after,
                'frame_idxs': frame_idxs
        }

        if use_labels:
            diff_features[diff]['step_type_labels'] = step_type_labels

    return diff_order, diff_features


def convert_dataset(
        dataset_name,
        new_dataset_name=None,
        raw_data_name=None,
        thresh=None,
        model_name=None,
        base_path=utils.BASE_PATH):
    """
    datasets assumed to be stored in {base_path}/datasets/{dataset_name}.
    Output will be in {base_path}/datasets/{new_dataset_name}.

    If raw_data_name is points to a directory, then use the sm files for
    bpm estimation.

    If model is provided, then steps will be based off of predictions, and
    threshold will be estimated.
    """

    if new_dataset_name is None:
        new_dataset_name = f"{dataset_name}_gen_{__version__}"
    print(f"New dataset name: {new_dataset_name}")

    new_ds_path = f"{base_path}/datasets/{new_dataset_name}"
    if not os.path.isdir(new_ds_path):
        os.mkdir(new_ds_path)

    smds = SMDUtils.get_dataset_from_file(
            dataset_name, 'placement', chunk_size=-1, concat=False)

    # Compute threshold if model is provided.
    if model_name is not None:
        model = StepPlacement.RegularizedRecurrentStepPlacementModel()
        model.load_state_dict(torch.load(model_name))
        model.cuda()

        thresh_ds = datautils.ConcatDataset(smds[:10])
        thresh = compute_thresh(model, thresh_ds)

    # Get BPM estimations.
    for smd in smds:
        print(smd.song_name)
        n_diffs = len(smd)

        if raw_data_name is None:
            # Compute BPM.
            pos_labels = []
            for d in smd:
                pos_labels.append(d['step_pos_labels'])

            pos_labels = np.concatenate(pos_labels)

            # For training, use ground truth step positions.
            bpm = bpm_estimator.est_bpm(pos_labels)
        else:
            sm = SMData.SMFile(smd.song_name, raw_data_name, base_path)
            try:
                bpm = bpm_estimator.true_bpm(sm)
            except ValueError as e:
                print(e)
                print(f"Skipping song {smd.song_name}")
                continue

        bps = 60 / bpm # Seconds per beat

        frame_idxs = None
        if model_name is not None:
            predict_loader = datautils.DataLoader(smd)
            outputs_list, labels_list = model.predict(
                    predict_loader, return_list=True)
            outputs_list = list(map(lambda l: l[0,:,0], outputs_list))
            labels_list = list(map(lambda l: l[0, :], labels_list))

            frame_idxs = list(map(
                lambda outputs: np.where(outputs > thresh)[0],
                outputs_list))

        diff_order, diff_features = get_generation_features(
                smd, bpm, frame_idxs)

        song_path = f'{new_ds_path}/{smd.song_name}'
        fname = f'{song_path}/{smd.song_name}.h5'

        if not os.path.isdir(song_path):
            os.mkdir(song_path)

        with h5py.File(fname, 'w') as hf:
            hf.attrs['song_name'] = smd.song_name
            hf.attrs['diff_names'] = np.array(diff_order).astype('S9')

            for diff in diff_order:
                diff_group = hf.create_group(diff)
                diff_data = diff_features[diff]

                for key in diff_data.keys():
                    diff_group.create_dataset(key, data=diff_data[key])

    return new_dataset_name
