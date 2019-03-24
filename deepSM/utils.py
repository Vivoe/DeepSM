import subprocess
import itertools
import glob
import requests
import json
import os
import importlib
import time

from deepSM import wavutils


BASE_PATH = '/home/lence/dev/deepStep'

def timestamp():
    os.environ['TZ'] = 'America/New_York'
    time.tzset()
    ts = time.strftime('%Y-%m-%d_%H-%M-%S')
    return ts

def format_time(s):
    hours, rem = divmod(s, 3600)
    minutes, secs = divmod(rem, 60)

    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), secs)

def convert_to_mono():
    mp3s = glob.glob('data/*/*.mp3')
    for mp3 in mp3s:
        fname = mp3.split('.')[0]
        subprocess.call(['ffmpeg', '-y', '-i', fname+'.mp3', '-ac', '1', fname+'.wav'])

def flatmap(a):
    return list(itertools.chain.from_iterable(a))

def inv_dict(d):
    return dict(map(lambda x: (x[1], x[0]), d.items()))

difficulties = {
    'Beginner':0,
    'Easy':1,
    'Medium':2,
    'Hard':3,
    'Challenge':4
}

inv_difficulties = inv_dict(difficulties)

def log_run(env):
    data = {}

    fields = [
            'model_name',
            'model_save',
            'train_ts',
            'dataset_name',
            'train_dataset_name',
            'test_dataset_name',
            'chunk_size',
            'n_songs',
            'fft_shape',
            'final_loss',
            'accuracy',
            'roc',
            'prauc',
            'f1'
    ]

    for field in fields:
        data[field] = env.get(field)

    print(data)

    with open('/home/ubuntu/dev/deepStep/models.log', 'a') as f:
        json.dump(data, f)
        f.write('\n')

def notify(message):
    request_url = 'https://api.ngjustin.com/notify/sendNotif'
    key = os.environ['JNG_KEY']

    headers = {'x-api-key': key}
    data = json.dumps({'message': str(message)})
    req = requests.post(request_url, headers=headers, data=data)

    if not req.ok:
        print("Status code:", req.status_code)
        print(req.text)


if importlib.util.find_spec("IPython"):
    from IPython.core.display import display, HTML

    def play_wav_jupyter(filepath):



        src = """
        <head>
        <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
        <title>Play wav</title>
        </head>

        <body>
        <audio controls="controls" style="width:600px">
            <source src="files/%s" type="audio/wav" />
            Your browser does not support the audio element.
        </audio>
        </body>
        """ % filepath

        display(HTML(src))

    def play_wav(data):
        if not isinstance(data, str):
            if not os.path.isdir('temp'):
                os.mkdir('temp')
            if os.path.isfile('temp/playwav.wav'):
                print("RM")
                os.remove('temp/playwav.wav')
            wavutils.write_wav('temp/playwav.wav', data)
            play_wav('temp/playwav.wav')

        else:
            play_wav_jupyter(data)

