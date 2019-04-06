import os
import argparse

import numpy as np
from sklearn import metrics

import matplotlib.pyplot as plt

import torch
import torch.utils.data as datautils

from deepSM import StepPlacement
from deepSM import SMDUtils
from deepSM import post_processing
from deepSM import utils


parser = argparse.ArgumentParser()
parser.add_argument('placement_model', type=str)
parser.add_argument('dataset_name', type=str)
parser.add_argument('--n_batches', type=int, default=2000)
parser.add_argument('--chunk_size', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)

args = parser.parse_args()

print("Testing model", args.placement_model)
print("Datset name:", args.dataset_name)

test_dataset = SMDUtils.get_dataset_from_file(
        args.dataset_name + '_test',
        'placement',
        chunk_size=args.chunk_size)

test_loader = datautils.DataLoader(
        test_dataset,
        num_workers = 4,
        batch_size = args.batch_size)

model = StepPlacement.RegularizedRecurrentStepPlacementModel()
model.load_state_dict(torch.load(args.placement_model))
model.cuda()

outputs, labels = model.predict(test_loader, max_batches=args.n_batches)

pmodel_str = args.placement_model.split('/')[-1][:-3]
torch.save(outputs, f'outputs_{args.dataset_name}_{pmodel_str}.torch')

def evaluate(outputs, labels):

    def zscore(x):
        return (x - x.mean()) / np.std(x)

    preds = zscore(outputs) > 1.5
    acc = metrics.accuracy_score(labels, preds)
    print("Accuracy:", acc)

    print("Percent positive:", preds.mean())

    roc = metrics.roc_auc_score(labels, outputs)
    print("ROC-AUC:", roc)

    precision, recall, thresh = metrics.precision_recall_curve(labels, outputs)

    prauc = metrics.auc(recall, precision)
    print("PR-AUC:", prauc)

    f1 = metrics.f1_score(labels, preds)
    print("F1 score:", f1)

print("Smoothed preds results:")
smoothed_outputs = post_processing.smooth_outputs(outputs)
evaluate(smoothed_outputs, labels)

print("Naive preds results:")
evaluate(outputs, labels)


utils.notify("DONE")
