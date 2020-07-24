import argparse
import numpy as np
import os
import shutil

from deepSM.utils import BASE_PATH

def split_dataset(ds_name, val_split, test_split):
    ds_path = f"{BASE_PATH}/data/processed/{ds_name}"
    output_path = f"{BASE_PATH}/data/development/{ds_name}"

    if not os.path.exists(f"{BASE_PATH}/data/development"):
        os.mkdir(f"{BASE_PATH}/data/development")
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    song_names = os.listdir(ds_path)
    np.random.shuffle(song_names)

    n_test = int(np.round(len(song_names) * test_split))
    n_val = int(np.round(len(song_names) * val_split))
    n_train = len(song_names) - n_test - n_val
    
    print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")

    train_songs = song_names[:n_train]
    val_songs = song_names[n_train:-n_test]
    test_songs = song_names[-n_test:]

    train_path = f"{output_path}/train"
    val_path = f"{output_path}/validation"
    test_path = f"{output_path}/test"

    if not os.path.isdir(train_path):
        os.mkdir(train_path)
    if not os.path.isdir(test_path):
        os.mkdir(test_path)
    if not os.path.isdir(val_path) and val_split > 0:
        os.mkdir(val_path)

    for song in train_songs:
        shutil.copytree(f"{ds_path}/{song}", f"{train_path}/{song}")

    for song in test_songs:
        shutil.copytree(f"{ds_path}/{song}", f"{test_path}/{song}")

    if val_split > 0:
        for song in val_songs:
            shutil.copytree(f"{ds_path}/{song}", f"{val_path}/{song}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--val_split", type=float, default=0)
    parser.add_argument("--test_split", type=float, default=0.25)
    args = parser.parse_args()

    split_dataset(args.dataset, args.val_split, args.test_split)