import argparse
import os
import glob
import shutil
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument('dataset_name', type=str)

args = parser.parse_args()

dataset_name = args.dataset_name
if dataset_name.endswith('/'):
    dataset_name = dataset_name[:-1]

new_dataset_name = dataset_name + '_export'
print(new_dataset_name)
shutil.copytree(dataset_name, new_dataset_name)

os.chdir(new_dataset_name)

songs = os.listdir()

for song in songs:
    smfile = next(filter(lambda x: x.endswith('.sm'), os.listdir(song)))
    wavfile = next(filter(lambda x: x.endswith('.wav'), os.listdir(song)))
    # Point the training .sm file to an ogg file.
    subprocess.call([
        'sed', '-i', 's/wav/ogg/g', f'{song}/{smfile}'])

    # Create the ogg file if it does not already exist.
    ogg_name = wavfile[:-4] + '.ogg'

    if ogg_name not in os.listdir(song):
        subprocess.call([
            'ffmpeg', '-i', f'{song}/{wavfile}', f'{song}/{ogg_name}'
        ])
    else:
        print("Ogg found. ignoring.")

    # Remove all non-required files.
    songfiles = os.listdir(song)

    songfiles = filter(
            lambda x: not x.endswith('.sm') and not x.endswith(ogg_name),
            songfiles)

    for f in songfiles:
        os.remove(f'{song}/{f}')


os.chdir('../')
subprocess.call(['zip', '-r', f'{dataset_name}.zip', new_dataset_name])

shutil.rmtree(new_dataset_name)
