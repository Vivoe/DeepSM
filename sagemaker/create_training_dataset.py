import argparse
import io
import glob
import os
import numpy as np
from scipy.io import wavfile
from subprocess import Popen, PIPE
import subprocess
import shutil
import uuid

from deepSM import SMData
import deepSM.beat_time_converter as BTC
from deepSM import SMDataset
from deepSM import utils, wavutils


def model_fn(model_dir):
    print("Loading model")
    print("May want to store hyperparameters in the model fn?")
    return None


def input_fn(body, content_type):
    # Expects zip file.
    print(f"Input function: content type: {content_type}. Body type: {type(body)}")
    
    init_dir = os.getcwd()
    
    temp_dir = str(uuid.uuid1())
    print("UUID", temp_dir)
    
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
        
    os.mkdir(temp_dir)
    with open(f'{temp_dir}/tempfile.zip', 'wb') as f:
        f.write(body)
        
    p = Popen(['unzip', f'{temp_dir}/tempfile.zip', '-d', temp_dir], stdout=PIPE, stderr=PIPE)
    print(f'UNZIP {p.communicate()}')
    
    p = Popen(['rm', f'{temp_dir}/tempfile.zip'], stdout=PIPE, stderr=PIPE)
    p.communicate()
    
    print(f'LISTDIR TEMP_DIR', os.listdir(temp_dir))
    new_dir = os.listdir(temp_dir)[0]
    
    print("NEW DIR", new_dir)
    
    print("TEMP_DIR/NEW_DIR", os.listdir(f'{temp_dir}/{new_dir}'))
    
    # Expects audio file and .sm file. Load both into memory.
    # Find audio file name.
    files = os.listdir(f'{temp_dir}/{new_dir}')
    audiofilename = next(filter(lambda x: x[-4:].lower() in ['.mp3', '.ogg', '.wav'], files))
    
    p = Popen(['ffmpeg', '-y',
        '-i', f'{temp_dir}/{new_dir}/{audiofilename}',
        '-ac', '1',
        '-ar', '44100',
        f'{temp_dir}/{new_dir}/tempaudio.wav'
    ], stdout=PIPE, stderr=PIPE)
    
    print("FFMPEG", p.communicate())
    
    _, data = wavfile.read(f'{temp_dir}/{new_dir}/tempaudio.wav')
    data = data / 32767
    
    # Read smfile into memory.
    smfilename = next(filter(lambda x: x[-3:] == '.sm', files))
    songname = smfilename[:-3]
    with open(f'{temp_dir}/{new_dir}/{smfilename}') as f:
        smfile = f.read()
        
    os.chdir(init_dir)
    shutil.rmtree(temp_dir)
    
    print("UUID DONE", temp_dir)
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
    fft_features = wavutils.gen_fft_features(padded_wav)

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


    smd = SMDataset.SMDataset(
            songname, diffs, fft_features, step_pos_labels, step_type_labels)
    
    return smd

def output_fn(data, ret_content_type):
    print("OUTPUTFUNCTION")
    
    buf = io.BytesIO()
    buf = data.save(buf)
    
    return buf.getvalue()



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    args = parser.parse_args()
    
    with open(f'{args.model_dir}/model.joblib', 'w') as f:
        f.write('Justin is stoopid')