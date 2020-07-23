import argparse
import numpy as np
import os
import shutil

from deepSM.utils import BASE_PATH

def split_dataset(ds_name, test_split):
    ds_path = f"{BASE_PATH}/data/processed/{ds_name}"
    output_path = f"{BASE_PATH}/data/development/{ds_name}"

    if not os.path.exists(f"{BASE_PATH}/data/development"):
        os.mkdir(f"{BASE_PATH}/data/development")
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    song_names = os.listdir(ds_path)
    np.random.shuffle(song_names)

    n_train = int(np.round(len(song_names) * (1 - test_split)))
    train_songs = song_names[:n_train]
    test_songs = song_names[n_train:]

    train_path = f"{output_path}/train"
    test_path = f"{output_path}/test"
    if not os.path.isdir(train_path):
        os.mkdir(train_path)
    if not os.path.isdir(test_path):
        os.mkdir(test_path)

    for song in train_songs:
        shutil.copytree(f"{ds_path}/{song}", f"{train_path}/{song}")

    for song in test_songs:
        shutil.copytree(f"{ds_path}/{song}", f"{test_path}/{song}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--test_split", type=float, default=0.25)
    args = parser.parse_args()

    split_dataset(args.dataset, args.test_split)