import os
import argparse
import numpy as np
import json

from deepSM import SMDUtils
from deepSM import utils
from deepSM import StepPlacement
from deepSM import post_processing

import torch
import torch.utils.data as datautils


parser = argparse.ArgumentParser()

parser.add_argument('dataset_name', type=str)
parser.add_argument('placement_model', type=str)

args = parser.parse_args()


smds = SMDUtils.get_dataset_from_file(
        args.dataset_name,
        'placement',
        chunk_size=-1,
        concat=False)

diffs = ['Beginner', 'Easy', 'Medium', 'Hard', 'Challenge']
thresholds = {}
for diff in diffs:
    thresholds[diff] = []

targets = dict(list(zip(diffs, [50, 66, 130, 220, 380])))

model = StepPlacement.RegularizedRecurrentStepPlacementModel()
model.load_state_dict(torch.load(args.placement_model))
model.cuda()

for smd in smds:
    print("Loading song", smd.song_name)
    loader = datautils.DataLoader(smd)

    outputs_list, labels_list = model.predict(loader, return_list=True)

    for outputs, labels, diff in zip(outputs_list, labels_list, smd.diffs):
        print(diff)
        outputs = outputs.reshape(-1).numpy()
        labels = labels.reshape(-1).numpy()
        smoothed = post_processing.smooth_outputs(outputs)

        thresh = post_processing.optimize_threshold_count(
                smoothed, labels, targets[diff])

        thresholds[diff].append(thresh)

res = {}
for diff in diffs:
    if len(thresholds[diff]) == 0:
        res[diff] = 2
    else:
        res[diff] = np.median(thresholds[diff])

model_name = args.placement_model.split('/')[-1][:-3]
out_path = f'{utils.BASE_PATH}/thresholds/count_{args.dataset_name}_{model_name}.json'
with open(out_path, 'w') as f:
    json.dump(res, f)



