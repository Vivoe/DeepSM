"""
Processes raw song packs into a more usable format.
Converts the .mp3/.ogg files into a single channel wav file,
and modifies the sm file to point to the wav file.
"""

import os
import argparse
import subprocess


def preprocess_dataset(ds_name):
    songs = os.listdir(f'data/{ds_name}')

    for song in songs:
        print(song)
        song_dir = f'data/{ds_name}/{song}'
        files = os.listdir(song_dir)
        sm_file = next(filter(lambda x: x.endswith('.sm'), files))
        audio_file = next(filter(
            lambda x: x[-4:] == '.ogg' or x[-4:] == '.mp3',
            files))

        print(audio_file)

        subprocess.call([ 'ffmpeg', '-y',
            '-i', f'{song_dir}/{audio_file}',
            '-ac', '1',
            '-ar', '44100',
            f'{song_dir}/{audio_file[:-4]}.wav'
        ])

        subprocess.call([
            'sed', '-i',
            f's/#MUSIC.*/#MUSIC:{audio_file[:-4]}.wav;/g',
            f'{song_dir}/{sm_file}'
        ])



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description=
# """
# Prepares .sm and audio files for training.
# Converts audio files to single channel .wav files, and directs
# the .sm files to the new .wav files.
# """)

#     parser.add_argument('dataset', type=str, help='Directory of song files.')

#     args = parser.parse_args()

#     preprocess_dataset(args.dataset)

