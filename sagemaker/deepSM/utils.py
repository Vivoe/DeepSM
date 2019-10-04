import subprocess
import itertools
import glob
import requests
import json
import os
import importlib
import time

import boto3

utils_path = os.path.dirname(os.path.realpath(__file__))
BASE_PATH = '/'.join(utils_path.split('/')[:-1])

def timestamp():
    os.environ['TZ'] = 'America/New_York'
    time.tzset()
    ts = time.strftime('%Y-%m-%d_%H-%M-%S')
    return ts

def format_time(s):
    hours, rem = divmod(s, 3600)
    minutes, secs = divmod(rem, 60)

    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), secs)

def convert_to_mono(dataset):
    mp3s = glob.glob(f'data/{dataset}/*/*.mp3')
    oggs = glob.glob(f'data/{dataset}/*/*.ogg')
    audio_files = mp3s + oggs
    print(audio_files)

    for audiof in audio_files:
        print(audiof)
        fname = audiof[:-4]
        subprocess.call(['ffmpeg', '-y', '-i', audiof, '-ac', '1', '-ar', '44100', fname+'.wav'])

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

def my_deserializer(ret, y):
    buf = io.BytesIO(ret.read())
    hf = h5py.File(buf)
    return hf

def notify(msg):
    lambd = boto3.client('lambda', region_name='us-east-2')
    json_str = '{"message": "'+msg+'"}'
    lambd.invoke(FunctionName='sendNotif', 
                 Payload=json_str.encode(),
                 InvocationType='Event')
    
    
def send_image(img_url):
    lambd = boto3.client('lambda', region_name='us-east-2')
    data = {'image': img_url}
    lambd.invoke(FunctionName='sendNotif', 
                 Payload=json.dumps(data).encode(), 
                 InvocationType='Event')
    
    
def upload_image_obj(obj, bucket, path):
    s3 = boto3.client('s3')
    s3.upload_fileobj(obj, bucket, path,
                      ExtraArgs={'ACL':'public-read',
                                'ContentType':'image/png'})
    
    
def get_s3_url(bucket, path, region='us-west-1'):
    return f'https://{bucket}.s3-{region}.amazonaws.com/{path}'