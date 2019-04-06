import os
import glob
import argparse
import numpy as np

from deepSM import SMData
from deepSM import bpm_estimator

def evaluate_bpm(raw_data_path, gen_path):

    songs = os.listdir(gen_path)

    true_bpms = []
    est_bpms = []
    for song_name in songs:
        est_sm = SMData.SMFile(
                song_name, raw_data_path=raw_data_path + '/' + song_name)
        true_sm = SMData.SMFile (
                song_name, raw_data_path=gen_path + '/' + song_name)

        try:
            true_bpm = bpm_estimator.true_bpm(true_sm)
            est_bpm = bpm_estimator.true_bpm(est_sm)

            true_bpms.append(true_bpm)
            est_bpms.append(est_bpm)
        except:
            continue

    true_bpms = np.array(true_bpms)
    est_bpms = np.array(est_bpms)

    print(true_bpms)
    print(est_bpms)
    print("N songs:", len(true_bpms))
    print("MSE:", np.mean((true_bpms - est_bpms)**2))
    print("MAE:", np.mean(np.abs(true_bpms - est_bpms)))
    print("Accuracy:", np.mean(true_bpms == est_bpms))
    print("Off by one errors:", np.mean(np.abs(true_bpms - est_bpms) == 1))
    print(np.abs(true_bpms - est_bpms))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_data_path', type=str)
    parser.add_argument('gen_path', type=str)

    args = parser.parse_args()

    evaluate_bpm(args.raw_data_path, args.gen_path)
