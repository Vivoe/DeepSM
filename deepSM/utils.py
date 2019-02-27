import subprocess
import itertools
import glob
import requests
import json
import os
import importlib

from deepSM import wavutils

def convert_to_mono():
    mp3s = glob.glob('data/*/*.mp3')
    for mp3 in mp3s:
        fname = mp3.split('.')[0]
        subprocess.call(['ffmpeg', '-y', '-i', fname+'.mp3', '-ac', '1', fname+'.wav'])

def flatmap(a):
    return list(itertools.chain.from_iterable(a))

difficulties = {
    'Beginner':0,
    'Easy':1,
    'Medium':2,
    'Hard':3,
    'Challenge':4
}

def log_run(env):
    data = {
        'model': env.get('model_name'),
        'model_save': env.get('model_save'),
        'train_ts': env.get('train_ts'),
        'dataset_name': env.get('dataset_name'),
        'chunk_size': env.get('chunk_size'),
        'n_songs': env.get('n_songs'),
        'batch_size': env.get('batch_size'),
        'n_epochs': env.get('n_epochs'),
        'fft_shape': env.get('fft_shape'), 
        'final_loss': env.get('final_loss'),
        'accuracy': env.get('accuracy'),
        'percent_pos': env.get('percent_pos'), 'roc': env.get('roc'),
        'prauc': env.get('prauc'),
        'f1': env.get('f1')
    }

    with open('/home/ubuntu/dev/deepStep/models.log', 'a') as f:
        json.dump(data, f)
        f.write('\n')

def notify(message):
    request_url = 'https://api.ngjustin.com/notify/sendNotif'
    key = os.environ['JNG_KEY']
    
    headers = {'x-api-key': key}
    data = json.dumps({'message': message})
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
            wavutils.write_wav('temp/playwav.wav', data)
            play_wav('temp/playwav.wav')
    
        else:
            play_wav_jupyter(data)

