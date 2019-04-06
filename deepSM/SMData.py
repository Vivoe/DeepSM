import re
import os
from deepSM import wavutils
from deepSM import utils


def split_beat_value_list(line):
    if line == '':
        return []

    # Splits lists of format beat=value,beat=value.
    entries = line.split(',')

    def split_beat_value(entry):
        beat, value = entry.split('=')

        # Beats can be floats!
        return (float(beat), float(value))

    return list(map(split_beat_value, entries))

def filter_comments(line):
    return re.sub('//.*\\n', '', line)



class Notes:
    """
    Container class for notes.
    May contain some data analysis functions, but no real processing.
    """

    def __init__(
        self, offset, bpms, stops, diff_name,
        diff_value, chart_type, notes):

        self.offset = offset
        self.bpms = bpms
        self.stops = stops
        self.diff_name = diff_name
        self.diff_value = diff_value
        self.chart_type = chart_type
        self.notes = notes


class SMFile:
    """
    Holds information about each song.
    Contains parsing utils, notes, and audio representations
        associated with the song.
    """

    def __init__(self, fname, raw_data_name=None, base_path=utils.BASE_PATH,
            raw_data_path=None):
        print(fname)
        self.raw_data_name = raw_data_name
        if raw_data_path is None:
            self.raw_data_path = f"{base_path}/data/{self.raw_data_name}/{fname}"
        else:
            self.raw_data_path = raw_data_path

        self.load_sm(fname, base_path)
        self.load_wav(base_path)

    def load_sm(self, fname, base_path):
        self.fname = fname

        sm_file_name = next(filter(
            lambda x: x.endswith('.sm'),
            os.listdir(self.raw_data_path)))

        sm_file_path = f"{self.raw_data_path}/{sm_file_name}"
        with open(sm_file_path) as f:
            lines = list(map(
                # lambda x: filter_comments(x.strip()).replace('\ufeff', ''),
                lambda x: filter_comments(x.strip()),
                f.read().split(';')))

        # Process header information and parse notes.
        # return lines
        self.note_charts = {}
        for line in lines:
            if line.startswith('#TITLE:'):
                self.title = line.split(':')[1]

            elif line.startswith('#MUSIC:'):
                self.music = line.split(':')[1]
                # self.music = line.split(':')[1].replace(' ', '_')

            elif line.startswith('#BPMS:'):
                bpmline = line.split(':')[1]

                self.bpms = split_beat_value_list(bpmline)

            elif line.startswith("#STOPS:"):
                stopsline = line.split(':')[1]
                self.stops = split_beat_value_list(stopsline)

            # elif line.startswith("#OFFSET:"):
            if 'OFFSET' in line:
                self.offset = float(line.split(":")[1])

            elif line.startswith("#NOTES:"):
                self.parse_notes(line)

    def parse_notes(self, line):
        header, chart_type, desc, diff_name, \
            diff_value, groove_radar, data = \
                list(map(lambda x: x.strip(), line.split(':')))

        if chart_type != 'dance-single':
            return

        def remove_mines(note):
            return note.replace("M", '0')

        note_data = list(map(
            lambda measure: list(map(remove_mines, measure.split())),
            data.split(',')))


        note = Notes(
            self.offset, self.bpms, self.stops,
            diff_name, diff_value, chart_type, note_data)

        self.note_charts[diff_name] = note

    def load_wav(self, base_path='.'):
        # Assumes that the wav file is already mono.
        filepath = base_path + '/data/' + self.fname + '/' + self.music
        filepath = f"{self.raw_data_path}/{self.music}"
        wav_filepath = filepath[:-3] + 'wav'
        rate, data = wavutils.read_wav(wav_filepath)
        self.rate = rate
        self.wavdata = data

