"""
Copies raw files for songs in a dataset to gen for testing.
"""


import os
import shutil
import argparse

from deepSM import utils


parser = argparse.ArgumentParser()
parser.add_argument('base_name', type=str)
parser.add_argument('dataset_name', type=str)
parser.add_argument('output_name', type=str)

args = parser.parse_args()


songs = os.listdir(args.dataset_name)

base_path = f'{utils.BASE_PATH}/data/{args.base_name}'
dst_path = f'{utils.BASE_PATH}/gen/{args.output_name}'
os.mkdir(dst_path)

for song in songs:
    shutil.copytree(f"{base_path}/{song}", f"{dst_path}/{song}")

