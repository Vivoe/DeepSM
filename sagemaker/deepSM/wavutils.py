import numpy as np
from scipy.io import wavfile
from python_speech_features.base import fbank
import os
import warnings


def read_wav(fname):
    # Converts to [-1, 1] range.
    rate, data = wavfile.read(fname)
    assert rate == 44100, f"WHY ARE YOU NOT USING 44.1K @ {fname}"
    return rate, data / 32767

def write_wav(fname, data, rate=44100):
    # Assumes that data is a numpy array.
    # Normalize to 0 dBfs.
    wavfile.write(fname, rate, (32767 * data / np.max(data)).astype(np.int16))

def play_wav(data):
    write_wav('temp/playwav.wav', data)
    os.system('play temp/playwav.wav')


def test_alignment(wav, times):
    """
    Creates a 'ding' at each time specified.
    """

    out = wav.copy()

    ts = np.arange(44100/2) / 44100
    f = 500
    ding = np.sin(2 * np.pi * ts * f) * np.exp(-15 * ts)

    for time in times:
        time = int(time * 44100)

        dinglen = min(22050, len(wav) - time)

        if dinglen <= 0:
            continue

        out[time:time+dinglen] += ding[:dinglen]

    return out


def pad_wav(first_frame, last_frame, wav, step=512):
    """
    Aligns the wav file and the frames to be correctly padded for fft processing.
    Assumes that max fft feature length is 4096!
    """

    N_samples = len(wav)

    # If we need frames from before the start of the song, pad.
    # Need 7 frames of space from first note.
    n_front_pad = -min((first_frame - 7) * step, 0)

    # Need 7 frames + (4096 samples=8 frames) of space after the last note.
    n_back_pad = max(0, (last_frame + 7 + 8) * step - N_samples)

    padded_wav = np.r_[np.zeros(n_front_pad), wav, np.zeros(n_back_pad)]

    # Returns the frame adjustment required for the frames due to front padding.
    # Add this to all frames.
    return (n_front_pad // step, padded_wav)


def gen_fft_features(wav, step=512, nfft=[2048,4096], n_bands=80, log=True):
    features = []
    # Ignoring warnings here.
    # Will warn about issues calculating MEL filters when nfft = 1024.
    # Causes a strange black band at band 5ish. Will ignore for now.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print("alive")
        for fft_size in nfft:
            print("FFTING")
            # mel_features is of shape [T, F]
            mel_features, mel_energy = fbank(
                wav, nfft=fft_size,
                samplerate=44100, nfilt=n_bands, winfunc=np.hamming,
                lowfreq=27.5, highfreq=8000.0,
                winstep=512/44100)
            print("post fft")

            if log:
                features.append(np.log10(mel_features + 1e-4))
            else:
                features.append(mel_features)
                
        print("still alive????")

    # Reutnrs shape [Channels, Time, Frequency]
    # return np.log10(np.stack(features))
    return np.stack(features)
