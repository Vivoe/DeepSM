import re
import os
from deepSM import utils


class Chart:
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

    def __init__(self, path):
        # Path to the song directory.
        self.path = path

        # defaults
        self.offset = 0
        self.bpms = [(0.0, 120.0)]
        self.stops = []

        file_name = next(filter(
            lambda x: x.endswith('.sm') or x.endswith('.ssc'),
            os.listdir(self.path)))

        file_type = file_name.split('.')[-1]
        file_path = f"{self.path}/{file_name}"
        if file_type == 'sm':
            self.load_sm(file_path)
        elif file_type == 'ssc':
            raise ValueError("Too lazy to support ssc just for A4A!")

        music_path = f"{path}/{self.music}"
        assert os.path.exists(music_path), f"{music_path} does not exist." 

    def load_sm(self, fpath):
        with open(fpath) as f:
            lines = list(map(
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
                self.parse_chart(line)


    def parse_chart(self, line):
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

        chart = Chart(
            self.offset, self.bpms, self.stops,
            diff_name, diff_value, chart_type, note_data)

        self.note_charts[diff_name] = chart


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