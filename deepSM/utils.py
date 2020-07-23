import subprocess
import itertools
import glob
import requests
import json
import os
import importlib
import time
import yaml


utils_path = os.path.dirname(os.path.realpath(__file__))
BASE_PATH = '/'.join(utils_path.split('/')[:-1])


def load_config():
    global config

    with open(f'{BASE_PATH}/config/config.yaml') as f:
        config = yaml.safe_load(f)

load_config()


def timestamp():
    os.environ['TZ'] = 'America/New_York'
    time.tzset()
    ts = time.strftime('%Y-%m-%d_%H-%M-%S')
    return ts

def format_time(s):
    hours, rem = divmod(s, 3600)
    minutes, secs = divmod(rem, 60)

    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), secs)

def format_neptune_params(config=config):
    def _format(d):
        res = {}
        for key in d:
            if isinstance(d[key], dict):
                sub_dict = _format(d[key])
                for sub_key in sub_dict:
                    res[f"{key}.{sub_key}"] = sub_dict[sub_key]
            else:
                res[key] = d[key]
        return res

    return _format(config)

def get_neptune_api_token():
    with open('/home/lence/.neptune') as f:
        api_token = f.readline().strip()
    return api_token

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


