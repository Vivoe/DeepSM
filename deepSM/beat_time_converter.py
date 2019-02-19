import numpy as np

def beat_timespan(beats, bpm):
    time_per_beat = 60 / bpm
    return beats * time_per_beat

def time_beatspan(times, bpm):
    return times * bpm / 60


class BeatTimeConverter:
    """
    Converts beats to time and vice versa.
    Also provides some formatting of notes given the calculations.
    """
    def __init__(self, offset, bpms, stops):
        self.offset = offset
        self.bpms = bpms
        self.stops = stops
        self.gen_beat_time_mapping()

    def beat_to_time(self, beats):
        beat_idxs = np.digitize(beats, self.beat_buckets) - 1
        n_beats = beats - self.beat_buckets[beat_idxs]

        return self.time_buckets[beat_idxs] +\
            beat_timespan(n_beats, self.bucket_bpm[beat_idxs])

    def time_to_beat(self, times):
        time_idxs = np.digitize(times, self.time_buckets) - 1
        n_seconds = times - self.time_buckets[time_idxs]

        return self.beat_buckets[time_idxs] +\
            time_beatspan(n_seconds, self.bucket_bpm[time_idxs])


    def gen_time_notes(self, notes):
        """
        Converts the SM note representation into a list of (time, notes).
        Stores the result in self.norm_notes.
        """

        norm_notes = [] # List of (time, note) pairs.

        for measure_idx, notes in enumerate(notes):
            N = len(notes)
            enumerated_notes = filter(lambda x: x[1] != '0000', enumerate(notes))
            measure_time_notes = map(
                lambda x: (self.beat_to_time(measure_idx * 4 + 4 * x[0] / N), x[1]),
                enumerated_notes)

            norm_notes.extend(measure_time_notes)

        return list(zip(*norm_notes))

    def align_to_frame(self, times, step=512):
        # Round to the nearest multiple of 512 samples.
        raw_frames = np.round(np.array(times) * 44100 / step).astype(int)

        return raw_frames

    def gen_beat_time_mapping(self):
        """
        Must run this befoer running beat_to_time or time_to_beat.
        Generates mapping between beat and time.
        """

        events = list(map(lambda x: (x[0], x[1], 'bpm'), self.bpms)) +\
                list(map(lambda x: (x[0], x[1], 'stop'), self.stops))

        events = sorted(events, key=lambda x:x[0])

        # Must have one BPM at beat 0.
        assert events[0][0] == 0 and events[0][2] == 'bpm',\
            "Need initial BPM."

        # Generate time buckets, where time per beat is constant within a bucket.
        # BPM of the time is specified by the bucket BPM.
        # Beat buckets define at what beat each BPM starts at.
        # Time_buckets[i] specify what time beat beat_buckets[i] is at.
        # To calculate time for any given beat, find the beat bucket index i, then
        #     time_bucket[i] + delta_beats * BPM_time
        beat_buckets = [0]
        time_buckets = [-self.offset]
        bucket_bpm = [events[0][1]]

        # First point is hardcoded.
        for i in range(1, len(events)):
            beat, val, event_type = events[i]

            if event_type == 'stop':
                # Stops occur directly after the beat. Needed for proper bucketing.
                beat_buckets.append(beat + 1e-6)
                time_buckets.append(time_buckets[-1] + val)
                bucket_bpm.append(bucket_bpm[-1])

            elif event_type == 'bpm':
                # Time since last event
                prev_beat = beat_buckets[-1]

                # Fix delta error for beats.
                if events[i-1][2] == 'stop':
                    prev_beat -= 1e-6

                delta_t = beat_timespan(beat - prev_beat, bucket_bpm[-1])

                beat_buckets.append(beat)
                time_buckets.append(delta_t + time_buckets[-1])
                bucket_bpm.append(val)

        self.beat_buckets = np.array(beat_buckets)
        self.time_buckets = np.array(time_buckets)
        self.bucket_bpm = np.array(bucket_bpm)