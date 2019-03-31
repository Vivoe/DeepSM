import os
import shutil
import numpy as np



def measure_times_to_notes(times, st_time, bpm):
    """
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
        """
        Returns the point of which x is closest to.
        """
        n = x.shape[0]
        return np.argmin(np.abs(
            np.tile(points, (n, 1)) - x.reshape((-1, 1))), axis=1)

    # Discretize notes into 192 timepoints.
    valid_divs = [192]
    for div in [4, 8, 12, 16, 24, 32, 48, 64, 192]:
        # Times for each note time for a given subdivision.
        # divs = np.linspace(0, measure_length, div+1)[:-1]
        delta = measure_length / div

        # The notes, rounded towards the div times.
        note_idxs = np.round(measure_times / delta).astype(int)
        notes = note_idxs * delta

        # notes = nearest_point_idx(measure_times, divs)

        # Max error from true time to rounded subdivision.
        # err = np.mean(np.abs(measure_times - divs[notes]))

        err = np.mean(np.abs(measure_times - notes))

        # Max error no greater than 1.5 frames, and
        # last note does not wrap to next measure.
        if np.max(note_idxs) < div and np.max(err) <= 1.5 * time_per_frame:
            valid_divs.append(div)


    best_div = np.min(valid_divs)
    divs = np.linspace(0, measure_length, best_div+1)[:-1]
    notes = nearest_point_idx(measure_times, divs)

    return best_div, notes




def frames_to_measures(frames, bpm, offset=None, n_offsets=200):
    """
    Converts frame-times into measure-aligned SM format.
    If offset is None, estimate the offset.
    """

    time_per_frame = 512/44100
    measure_length = 4 * 60 / bpm # in seconds.
    n_measures = int(len(frames) * time_per_frame / measure_length) + 1

    note_times = np.where(frames == 1)[0] * time_per_frame

    def get_measure_notes(offset):
        """
        Converts frames measure-note format for a given offset.
        """
        measure_cut_times = np.arange(n_measures) * measure_length - offset
        measure_bkts = np.digitize(note_times, measure_cut_times) - 1

        def get_measure_div(measure_idx):
            """
            Gets the best subdivision, and the note indexes for a measure.
            """
            # Time for each note in the measure.
            times = note_times[measure_bkts == measure_idx]
            return measure_times_to_notes(
                    times, measure_cut_times[measure_idx], bpm)

        divnotes = list(map(get_measure_div, range(n_measures)))
        divnotes = list(zip(*divnotes))
        return divnotes

    if offset is None:
        # Finding optimal offset.
        offset_divs = np.ones(n_offsets) * np.inf
        offsets = np.linspace(0, measure_length, n_offsets)
        for i, offset in enumerate(offsets):
            divs = get_measure_notes(offset)[0]
            # Sum of all subdivions for the song with a given offset.
            offset_divs[i] = np.sum(divs)

        # Select the offset with the lowest sum of divisions.
        offset = offsets[np.argmin(offset_divs)]

    divnotes = get_measure_notes(offset)
    return offset, divnotes


def to_SMFile(title, music, diffs, offset, bpm,
        diff_divnotes, diff_steps, subtitle='',
        sm_path=None):
    """
    divnotes are the output from frames_to_measures, a tuple of (div, note_idx)
    steps is a n_notes x 4 matrix.
    """

    header_template = f"""
#TITLE:{title};
#SUBTITLE:{subtitle};
#ARTIST:;
#MUSIC:{music};
#OFFSET:{offset};
#SAMPLESTART:20.000;
#SAMPLELENGTH:10.000;
#SELECTABLE:YES;
#DISPLAYBPM:{bpm:.3f};
#BPMS:0.000={bpm:.3f};
#STOPS:;
    """


    notes_str = ''
    for diff, divnotes, steps in zip(diffs, diff_divnotes, diff_steps):
        print("Diff:", diff)

        notes_header_template = f"""
#NOTES:
    dance-single:
    DeepSM:
    {diff}:
    :
    :
"""

        notes_str += notes_header_template

        assert sum(map(len, divnotes[1])) == len(steps)
        notes = list(map(lambda line: ''.join(list(map(str, line))), steps))
        step_it = iter(notes)

        for div, note_pos in zip(*divnotes):
            notes = ['0000'] * div
            for pos in note_pos:
                notes[pos] = next(step_it)
            notes_str += '\n'.join(notes) + '\n,\n'

        notes_str = notes_str[:-2] + ';\n'

    if sm_path is None:
        sm_path = f'gen/{title}/{title}.sm'
        sm_song = f'gen/{title}/{title}.ogg' # Currently assumes always ogg.
        if not os.path.isdir(f'gen/{title}'):
            os.mkdir(f'gen/{title}')

        if os.path.isfile(sm_path):
            os.remove(sm_path)

        if not os.path.isfile(sm_song):
            shutil.copy(f'data/fraxtil-ext/{title}/{title}.ogg', sm_song)

    print("Writing .sm file to ", os.path.abspath(sm_path))

    with open(sm_path, 'w') as f:
        f.write(header_template)
        f.write(notes_str)
