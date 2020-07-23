import numpy as np
import torch
import torchaudio

import os
import warnings

from deepSM import utils

audio_config = utils.config['audio']


def load_raw_audio(fpath):
    
    audio_data, sr = torchaudio.load(fpath)
    assert sr == audio_config['sampleRate']
    return audio_data


def fft_processing(audio_data):
    # Use CPU loading as extra CPU bandwidth as opposed to GPU.
    mels = []
    for nfft in audio_config['fft']['nfft']:
        mel_trans = torchaudio.transforms.MelSpectrogram(
            n_fft=nfft,
            sample_rate=audio_config['sampleRate'],
            hop_length=audio_config['samplesPerFrame'],
            f_min=audio_config['fft']['fmin'],
            f_max=audio_config['fft']['fmax'],
            n_mels=audio_config['fft']['nMels']
        )

        mel = mel_trans(audio_data)
        mels.append(mel)

    # Dimensions are:
    # n_ffts x n_mels x n_frames
    mel_data = torch.cat(mels)
    return mel_data


def pad_audio(mel_data, first_frame, last_frame):
    # Happens after FFT. Should not change much?
    n_frames = mel_data.shape[-1]

    first_req_idx = first_frame - audio_config['contextWindowSize']
    n_front_pad = -min(first_req_idx, 0)

    last_req_idx = last_frame + audio_config['contextWindowSize']
    n_back_pad = max(last_req_idx - n_frames, 0)

    mel_data = torch.nn.functional.pad(
        mel_data, (n_front_pad, n_back_pad)
    )
    
    return mel_data, n_front_pad
