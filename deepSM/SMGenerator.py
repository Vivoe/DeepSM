#
# Converts SMFile objects and a list of frames into a .sm file.
#

import numpy as np

from scipy import signal
from scipy import stats
from statsmodels.tsa.stattools import acf

def est_bpm(smd, min_bpm=120):
    #
    # Uses autocorrelation and peak picking of note placements.
    #
            
    sm_notes = smd.step_pos_labels
    ests = [] 
    for i in range(sm_notes.shape[0]):
        notes_acf = acf(sm_notes[i,:], nlags=200)
        
        # Finds peaks.
        # Thresholds peaks based on height.
        # Grid searches for lowest height that minimizes the variance in the
        # distaces between peaks.
        # Helps reduce noise, particularly from changing BPM songs and
        # mixed triplets.
        min_var = np.inf
        height = None
        for i in np.linspace(1e-2, 0.5, 100):
            peaks = signal.find_peaks(notes_acf, height=i)[0]
            if len(peaks) <= 2:
                continue
            diffs = peaks[1:] - peaks[:-1]
            v = np.var(diffs)
            if v < min_var:
                min_var = v
                height = i

        peaks = signal.find_peaks(notes_acf, height=height)[0]
        diffs = peaks[1:] - peaks[:-1]

        if len(diffs) == 0:
            continue

        diff = np.mean(diffs) 

        # Mupltiples of two are essentially the same.
        # Choose the bpm bewteen [120, 240).
        time_per_diff = diff * 512/44100

        i = 0
        while True:
            secs_per_beat = time_per_diff * 2**i
            beats_per_sec = 1 / secs_per_beat
            bpm = beats_per_sec * 60

            if min_bpm <= bpm < 2 * min_bpm:
                break
            elif bpm >= 2 * min_bpm:
                i += 1
            elif bpm < min_bpm:
                    i -= 1
            else:
                print(bpm)
                raise ValueError("Something is wrong.")

        ests.append(bpm)
        
    # Get bpm estimate for each difficulty, return the consensus.
    modes = stats.mode(np.round(ests))
    return modes.mode


def bpm_error(frames, notes, bpm, offset):
    # Error based on MSE.
    # Frames should just be placement times.
    # Note format:
    # Nx3 array, 1st col for measure idx, 2nd for notes in measure, 3rd for pos
    # in measure
    # [(n_notes, [note_idxs])]
    time_per_frame = 512 / 44100
    frame_times = np.where(frames == 1)[0] * time_per_frame

    measure_length = 4 * 60 / bpm

    note_times = -offset + measure_length * \
            (notes[:,0] + notes[:,2] / notes[:,1])

    err = np.mean((note_times - frame_times)**2)
    return err


def generate_SMFile(smFile, frames):
    # How to decide BPM? Minimize error.    

    
