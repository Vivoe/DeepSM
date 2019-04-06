import os
import subprocess

songs = os.listdir()

for song in songs:
    smfile = next(filter(lambda x: x.endswith('.sm'), os.listdir(song)))
    subprocess.call([
        'sed', '-i', 's/wav/ogg/g', f'{song}/{smfile}'])


