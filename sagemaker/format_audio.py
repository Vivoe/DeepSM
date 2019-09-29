"""
Processes raw song packs into a more usable format.
Converts the .mp3/.ogg files into a single channel wav file,
and modifies the sm file to point to the wav file.
"""

import os
import argparse
import subprocess
from subprocess import Popen, PIPE

import re
from scipy.io import wavfile

import torch

def model_fn(model_dir):
    print("XXXXXXX Loading model...uh yeah")
    return None



def input_fn(body, req_content_type):
    # Take input as probably audio bytestream, write to file then output numpy wav.
        
    with open('tempaudio', 'wb') as f:
        f.write(body)

    p = Popen(['ffprobe', 'tempaudio'], stderr=PIPE)
    _, res = p.communicate()
    
    print(res)
    
    subprocess.call([ 'ffmpeg', '-y',
        '-i', 'tempaudio',
        '-ac', '1',
        '-ar', '44100',
        'tempaudio.wav'
    ])
    
    _, data = wavfile.read('tempaudio.wav')
    
    print(data.shape)
    
    return data


def predict_fn(data, model):
    return data

# For now, return numpy array. Use defualt serializer.

if __name__ == "__main__":
    print("MAIN")
    print("RELOADED2")
    print('"TRAINING" LOL')