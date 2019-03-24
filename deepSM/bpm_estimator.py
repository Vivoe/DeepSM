import numpy as np
from scipy import signal
from scipy import stats
from statsmodels.tsa.stattools import acf
import warnings

def cut_to_bpm_range(bpm, min_bpm=120):
    x = np.log2(bpm) - np.log2(min_bpm)
    return round(2**(x % 1 + np.log2(min_bpm)))

def est_bpm(frames, min_bpm=120):
    # frames: Array for each frame, 1 if step.
    notes_acf = acf(frames, nlags=200)

    min_var = np.inf
    height = None
    for i in np.linspace(1e-2, 0.5, 100):
        peaks = signal.find_peaks(notes_acf, height=i)[0]
        if len(peaks) <=  2:
            continue
        diffs = peaks[1:] - peaks[:-1]
        v = np.var(diffs)
        if v < min_var:
            min_var = v
            height = i

    peaks = signal.find_peaks(notes_acf, height=height)[0]
    diffs = peaks[1:] - peaks[:-1]
    diff = np.mean(diffs)


    # Get BPM est between [min_bpm, min_bpm*2)
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
            raise ValueError("Something went wrong in adjusting BPM.")

    return np.round(bpm).astype(int)


def true_bpm(sm, min_bpm=120, req_thresh=0.4):
    """
    Takes the BPM with the largest amount of time, or integer multiples of.
    Must compose of at least req_thresh% of the song.
    """

    n_beats = len(list(sm.note_charts.values())[0].notes) * 4
    bpms = sm.bpms

    bpms.append((n_beats, None))

    bpm_dict = {}
    for i in range(len(bpms)-1):
        bpm = cut_to_bpm_range(bpms[i][1], min_bpm)
        if bpm not in bpm_dict:
            bpm_dict[bpm] = 0

        bpm_dict[bpm] += bpms[i+1][0] - bpms[i][0]


    true_bpm = max(bpm_dict, key=bpm_dict.get)
    if bpm_dict[true_bpm] < req_thresh * n_beats:
        raise ValueError(f"Unable to get good estimate of BPM for song {sm.title}, probably non-static.")

    return true_bpm
