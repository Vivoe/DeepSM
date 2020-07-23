"""
Processes raw song packs into a more usable format.
Converts the .mp3/.ogg files into a single channel wav file,
and modifies the sm file to point to the wav file.
"""

import argparse
import os
import shutil
import sox
import subprocess

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from deepSM.utils import BASE_PATH 


def preprocess_dataset(ds_name):
    songs = os.listdir(f'{BASE_PATH}/data/Songs/{ds_name}')

    if not os.path.exists(f'{BASE_PATH}/data/processed'):
        os.mkdir(f'{BASE_PATH}/data/processed')

    output_dir = f'{BASE_PATH}/data/processed/{ds_name}'
    if os.path.exists(output_dir):
        # shutil.rmtree(output_dir)
        pass
    else:
        os.mkdir(output_dir)

    with ProcessPoolExecutor() as ex:
        list(ex.map(preprocess_song, zip(repeat(ds_name), songs)))

    print("Done.")

def preprocess_song(ds_song):
    ds_name, song = ds_song
    print(song)

    output_dir = f'{BASE_PATH}/data/processed/{ds_name}'

    song_dir = f'{BASE_PATH}/data/Songs/{ds_name}/{song}'
    if not os.path.isdir(song_dir):
        return

    files = os.listdir(song_dir)
    audio_file = next(filter(lambda x: x.endswith('.ogg') or x.endswith('.mp3'), files))
    sm_file = next(filter(lambda x: x.endswith('.sm') or x.endswith('.ssc'), files))

    song_output_dir = f'{output_dir}/{song}'

    if os.path.exists(song_output_dir):
        return
    os.mkdir(song_output_dir)

    sox_transform = sox.Transformer().convert(44100, 1)
    sox_transform.build(f'{song_dir}/{audio_file}', f'{song_output_dir}/{audio_file[:-4]}.mp3')

    shutil.copyfile(f'{song_dir}/{sm_file}', f'{song_output_dir}/{sm_file}')

    subprocess.call([
        'sed', '-i',
        f's/#MUSIC.*/#MUSIC:{audio_file[:-4]}.mp3;/g',
        f'{song_output_dir}/{sm_file}'
    ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
"""
Prepares .sm and audio files for training.
Converts audio files to single channel .wav files, and directs
the .sm files to the new .wav files.
""")

    parser.add_argument('dataset', type=str, help='Directory of song files.')
    args = parser.parse_args()
    preprocess_dataset(args.dataset)