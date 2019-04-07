import os
import shutil

def to_SMFile(title, music, diffs, offset, bpm,
        diff_divnotes, diff_steps, subtitle='',
        sm_path=None, comment=""):
    """
    divnotes are the output from frames_to_measures, a tuple of (div, note_idx)
    steps is a n_notes x 4 matrix.
    """

    header_template = f"""
//{comment}
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

        # Assertion can fail, since we now allow notes to overlap.
        # What will happen is that the latter overlapping note will take presidence.
        # assert sum(map(len, divnotes[1])) == len(steps)

        notes = list(map(lambda line: ''.join(list(map(str, line))), steps))
        step_it = iter(notes)

        for div, note_pos in zip(*divnotes):
            notes = ['0000'] * div
            for pos in note_pos:
                if pos >= div:
                    # Ignore note.
                    next(step_it)
                    continue

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
