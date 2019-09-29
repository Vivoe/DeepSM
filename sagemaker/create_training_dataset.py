import io
import glob
import os
import numpy as np
from scipy.io import wavfile
from subprocess import Popen, PIPE
import subprocess

from deepSM import SMData
import deepSM.beat_time_converter as BTC
from deepSM import SMDataset
from deepSM import utils, wavutils


def run_cmd(cmd):
    p = Popen(cmd, stdout=PIPE, stderr=PIPE)
    return p.communicate()

def model_fn(model_dir):
    print("Loading model")
    print("May want to store hyperparameters in the model fn?")
    return None


def input_fn(body, content_type):
    # Expects zip file.
    print("Input function.")
    print(content_type)
    print(type(body))
    os.mkdir('temp')
    os.chdir('temp')
    with open('tempfile.zip', 'wb') as f:
        f.write(body)
        
    p = Popen(['unzip', 'tempfile.zip'], stdout=PIPE, stderr=PIPE)
    print(p.communicate())
    
    p = Popen(['rm', 'tempfile.zip'], stdout=PIPE, stderr=PIPE)
    print(p.communicate())
    
    print(os.listdir())
    os.chdir(os.listdir()[0])
    
    print(os.listdir())
    
    # Expects audio file and .sm file. Load both into memory.
    # Find audio file name.
    files = os.listdir()
    audiofilename = next(filter(lambda x: x[-4:] in ['.mp3', '.ogg', '.wav'], files))
    
    print(run_cmd(['pwd']))
    
    p = Popen(['ffmpeg', '-y',
        '-i', audiofilename,
        '-ac', '1',
        '-ar', '44100',
        'tempaudio.wav'
    ], stdout=PIPE, stderr=PIPE)
    
    print(p.communicate())
    
    _, data = wavfile.read('tempaudio.wav')
    data = data / 32767
    
    # Read smfile into memory.
    smfilename = next(filter(lambda x: x[-3:] == '.sm', files))
    songname = smfilename[:-3]
    with open(smfilename) as f:
        smfile = f.read()
        
    return (songname, smfile, data)
    
    
def predict_fn(raw_data, model):
    # Convert raw wavdata and sm file to processed hdf5 format.
    
    songname, smdata, wavdata = raw_data
    
    sm = SMData.SMFile(songname, smdata, wavdata)
    btc = BTC.BeatTimeConverter(sm.offset, sm.bpms, sm.stops)
    
    # Get difficulties.
    diffs = list(filter(lambda x: x != 'Edit', sm.note_charts.keys()))
#     if drop_diffs is not None:
#         diffs = list(filter(lambda x: x not in drop_diffs, diffs))

    notes = {} # Contains only a list of notes for each difficulty.
    times = {} # List of times per diff.
    frames = {}
    # labels = {} # List of note aligned labels for note events. {0, 1} for now.


    # Track first and last notes for wav padding.
    first_frame = np.inf
    last_frame = -np.inf

    # Find note times and frames for alignment to features.
    # Will pad wavfile if first or last note is too close to beginning/end.
    for diff in diffs:
        times[diff], notes[diff] = \
            btc.gen_time_notes(sm.note_charts[diff].notes)

        frames[diff] = btc.align_to_frame(times[diff])

        if frames[diff][0] < first_frame:
            first_frame = frames[diff][0]
        if frames[diff][-1] > last_frame:
            last_frame = frames[diff][-1]

    front_pad_frames, padded_wav = \
            wavutils.pad_wav(first_frame, last_frame, sm.wavdata)


    # Get FFT Transform.
    fft_features = wavutils.gen_fft_features(padded_wav, log=True)

    
    return 'alive'
    # N_channels = 3 (1024, 2048, 4096)
    # N_frames ~ song length * 44100 / 512
    # N_freqs = 80 (Number of mel coefs per frame)
    N_channels, N_frames, N_freqs = fft_features.shape

    
    # Get labels (step position and type)
    step_pos_labels = np.zeros((len(diffs), N_frames))
    step_type_labels = np.zeros((len(diffs), N_frames, 4))
    for i, diff in enumerate(diffs):
        # Adjusting for the new frames added on to the front.
        frames[diff] += front_pad_frames

        step_pos_labels[i, frames[diff]] = 1

        for j, note in zip(frames[diff], notes[diff]):
            step_type_labels[i, j, :] = np.array(list(map(int, note)))


    return SMDataset.SMDataset(
            songname, diffs, fft_features, step_pos_labels, step_type_labels)

def output_fn(data, ret_content_type):
    # Return HDF5 bytestream.
    ret = data.save(io.BytesIO()).read()
    
    return ret