import numpy as np
from scipy import signal
from scipy import stats
from statsmodels.tsa.stattools import acf
import warnings

from deepSM import beat_alignment

def cut_to_bpm_range(bpm, min_bpm=120):
    x = np.log2(bpm) - np.log2(min_bpm)
    return round(2**(x % 1 + np.log2(min_bpm)))

def est_bpm(frames, min_bpm=120):
    # frames: Array for each frame, 1 if step.
    notes_acf = acf(frames, nlags=200)

    min_var = np.inf
    height = None
    for i in np.linspace(1e-2, 0.5, 20):
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
    # Equal to the spacing between each "note"
    time_per_diff = diff * 512/44100

    beats_per_sec = 1/time_per_diff
    bpm = beats_per_sec * 60

    bpm = cut_to_bpm_range(bpm)

    return np.round(bpm).astype(int)

def refined_bpm_estimate(preds, min_bpm=120):
    """
    Estimates BPM across all difficulties, and searches for the best one.
    Preds should be a dict, with diff_names as key and frames as value.
    """

    bpms = []
    for diff in preds.keys():
        bpm = est_bpm(preds[diff])
        bpms.append(bpm)

    candidate_bpms = []
    for bpm in bpms:
        candidate_bpms.append(bpm-1)
        candidate_bpms.append(bpm)
        candidate_bpms.append(bpm+1)

    candidate_bpms = list(set(candidate_bpms))
    print("Candidate BPMs:", candidate_bpms)

    bpm_scores = []
    for bpm in candidate_bpms:
        print("Processing", bpm)
        score = 0
        for diff in preds.keys():
            offset, divnotes = \
                    beat_alignment.frames_to_measures(preds[diff], bpm)

            score += sum(divnotes[0])

        bpm_scores.append(score)

    bpm_idx = np.argmin(bpm_scores)
    bpm = candidate_bpms[bpm_idx]
    print("BPM scores:", bpm_scores)
    return bpm


def true_bpm(sm, min_bpm=120, req_thresh=0.4):
    """
    Gets the fundamental BPM from the source file.

    Takes the BPM with the largest amount of time, or integer multiples of.
    Must compose of at least req_thresh% of the song.
    """

    n_beats = len(list(sm.note_charts.values())[0].notes) * 4
    bpms = sm.bpms.copy()

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

    true_bpm = cut_to_bpm_range(true_bpm)
    return true_bpm
