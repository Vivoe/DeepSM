import os
import numpy as np
from sklearn.metrics import f1_score
import h5py
import warnings

from deepSM import SMDUtils
from deepSM import StepPlacement
from deepSM import utils
from deepSM import bpm_estimator

from torch.utils import data as datautils

__version__ = '1-0'

from importlib import reload
reload(utils)


def compute_thresh(model, dataset):
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


def convert_dataset(
        dataset_name,
        thresh=None,
        base_path=utils.BASE_PATH):

    new_dataset_name = f"{dataset_name}_gen_{__version__}"
    print(f"New dataset name: {new_dataset_name}")

    new_ds_path = f"{base_path}/datasets/{new_dataset_name}"
    if not os.path.isdir(new_ds_path):
        os.mkdir(new_ds_path)

    smds = SMDUtils.get_dataset_from_file(
            dataset_name, chunk_size=-1, concat=False)

    for smd in smds:
        print(smd.song_name)
        n_diffs = len(smd)

        # Compute BPM.
        pos_labels = []
        for d in smd:
            pos_labels.append(d['step_pos_labels'])

        pos_labels = np.concatenate(pos_labels)

        # For training, use ground truth step positions.
        bpm = bpm_estimator.est_bpm(pos_labels)
        bps = 60 / bpm # Seconds per beat

        # Aggregate features.
        diff_order = []
        diff_features = {}
        for d in smd:
            # Mantain order of difficulties.
            diff_code = np.where(d['diff'][0])[0][0]
            diff = utils.inv_difficulties[diff_code] # String
            diff_order.append(diff)

            frame_idxs = np.where(d['step_pos_labels'])[0]
            fft_features = d['fft_features'][frame_idxs]
            step_type_labels = d['step_type_labels'][frame_idxs]

            # Compute delta_time, delta_beat
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
                    'step_type_labels': step_type_labels
            }

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



