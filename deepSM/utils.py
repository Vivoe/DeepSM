import subprocess
import itertools
import glob

def convert_to_mono():
    mp3s = glob.glob('data/*/*.mp3')
    for mp3 in mp3s:
        fname = mp3.split('.')[0]
        subprocess.call(['ffmpeg', '-y', '-i', fname+'.mp3', '-ac', '1', fname+'.wav'])

def flatmap(a):
    return list(itertools.chain.from_iterable(a))

difficulties = {
    'Beginner':0,
    'Easy':1,
    'Medium':2,
    'Hard':3,
    'Challenge':4
}