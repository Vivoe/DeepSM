"""
Finds the optimal offset and beat alignment given frames.
"""

import numpy as np


def measure_times_to_notes(times, st_time, bpm, drop_subdivs=None):
    """
    Aligns note times within a note to measure subdivisions.
    Selects the number of subdivisions appropriately.

    times: time for each note in the measure. Time relative to song start.
    st_time: Starting time of the measure.
    bpm: bpm.
    return: The subdivision, and the index of notes into the subdivisions.
    """

    if len(times) == 0:
        return 4, np.array([])

    measure_length = 4 * 60 / bpm
    time_per_frame = 512/44100

    # Times of notes within the measure.
    measure_times = times - st_time

    def nearest_point_idx(x, points):
        # Returns the point of which x is closest to.
        n = x.shape[0]
        return np.argmin(np.abs(
            np.tile(points, (n, 1)) - x.reshape((-1, 1))), axis=1)

    # Discretize notes into 192 timepoints.
    div_list = [4, 8, 12, 16, 24, 32, 48, 64, 192]
    if drop_subdivs:
        div_list = div_list[:-drop_subdivs]
        div_list = [4, 8, 12, 16, 24]

    valid_divs = [div_list[-1]]
    for div in div_list:
        # Times for each note time for a given subdivision.
        delta = measure_length / div

        # The notes, rounded towards the div times.
        note_idxs = np.round(measure_times / delta).astype(int)
        notes = note_idxs * delta

        # Max error from true time to rounded subdivision.
        err = np.mean(np.abs(measure_times - notes))

        # Max error no greater than 1.5 frames, and
        # last note does not wrap to next measure.
        if np.max(note_idxs) < div and np.max(err) <= 1.5 * time_per_frame:
            valid_divs.append(div)


    # Take the smallest valid measure.
    best_div = np.min(valid_divs)

    delta = measure_length / best_div
    notes = np.round(measure_times / delta).astype(int)
    # divs = np.linspace(0, measure_length, best_div+1)[:-1]
    # notes = nearest_point_idx(measure_times, divs)

    return best_div, notes



def frames_to_measures(frames, bpm, offset=None, n_offsets=200, drop_subdivs=None):
    """
    Converts frame-times into measure-aligned SM format.
    If offset is None, estimate the offset.
    """

    time_per_frame = 512/44100
    measure_length = 4 * 60 / bpm # in seconds.
    n_measures = int(len(frames) * time_per_frame / measure_length) + 3

    note_times = np.where(frames == 1)[0] * time_per_frame


    def get_measure_notes(offset, drop_subdivs=None):
        """
        Converts frames measure-note format for a given offset.
        """
        measure_cut_times = np.arange(n_measures) * measure_length - offset
        measure_bkts = np.digitize(note_times, measure_cut_times)

        def get_measure_div(measure_idx):
            """
            Gets the best subdivision, and the note indexes for a measure.
            """
            # Time for each note in the measure.
            times = note_times[measure_bkts == measure_idx]
            res = measure_times_to_notes(
                    times, measure_cut_times[measure_idx], bpm,
                    drop_subdivs)
            assert len(times) == len(res[1])
            return res


        divnotes = list(map(get_measure_div, range(n_measures)))
        divnotes = list(zip(*divnotes))
        return divnotes


    if offset is None:
        # Finding optimal offset.
        offset_divs = np.ones(n_offsets) * np.inf
        offsets = np.linspace(-measure_length, measure_length, n_offsets)
        for i, offset in enumerate(offsets):
            divs = get_measure_notes(offset)[0]
            # Sum of all subdivions for the song with a given offset.
            offset_divs[i] = np.sum(divs)

        # Select the offset with the lowest sum of divisions.
        offset = offsets[np.argmin(offset_divs)]

    divnotes = get_measure_notes(offset, drop_subdivs)
    return offset, divnotes

