import os
import time
import re
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, f1_score

from deepSM import SMDataset
from deepSM import SMDUtils
from deepSM import StepPlacement
from deepSM import utils
from deepSM import samplers

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data as datautils

assert os.getcwd().split('/')[-1] == 'deepStep'

parser = argparse.ArgumentParser()
parser.add_argument('dataset_name', type=str)
parser.add_argument('model_name', type=str)
parser.add_argument('--chunk_size', type=int, default=100)
parser.add_argument('--n_songs', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_epochs', type=int, default=1)
parser.add_argument('--init_weights', type=str)

args = parser.parse_args()

dataset_name = args.dataset_name
model_name = args.model_name
chunk_size = args.chunk_size
n_songs = args.n_songs
n_epochs = args.n_epochs
batch_size = args.batch_size
init_weights = args.init_weights

train_dataset, train_step_pos_labels = SMDUtils.get_dataset_from_file(
    dataset_name + '_train', 'placement', n_songs=n_songs, chunk_size=chunk_size, step_pos_labels=True)

test_dataset, test_step_pos_labels = SMDUtils.get_dataset_from_file(
    dataset_name + '_test', 'placement', n_songs=n_songs, chunk_size=chunk_size, step_pos_labels=True)

print(np.mean(train_step_pos_labels))
print(np.mean(test_step_pos_labels))

# Train/test sets are pre-generated.
train_loader = datautils.DataLoader(
    train_dataset,
    num_workers=8,
    batch_size = batch_size,
#     sampler=samp,
    shuffle=True,
    pin_memory=True)
print(len(train_dataset))

test_loader = datautils.DataLoader(
    test_dataset, num_workers=8,
    shuffle=True,
    batch_size=batch_size, pin_memory=True)

print(len(test_dataset))


train_ts = utils.timestamp()

# model = StepPlacement.RegularizedConvStepPlacementModel(log=True)
model = StepPlacement.RegularizedRecurrentStepPlacementModel()

if init_weights:
    print(f"Initializing weights from {init_weights}")
    model.load_state_dict(torch.load(init_weights))
model.cuda()

model.fit(train_loader, n_epochs, batch_size)

outputs, labels = model.predict(test_loader, max_batches=2000)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

preds = sigmoid(outputs) > 0.1

accuracy = 1 - np.mean(np.abs(preds - labels))
print("Accuracy:", accuracy)

percent_pos = np.mean(preds)
print("Percent positive: ", percent_pos)

roc = roc_auc_score(
    labels,
    outputs)

print("ROC:", roc)

precision, recall, thresh = precision_recall_curve(
    labels,
    outputs)

prauc = auc(recall, precision)
print("AUC PR:", prauc)

f1 = f1_score(labels, preds)
print("F1 score:", f1)

utils.notify(f"Acc: {accuracy}, PPos: {percent_pos}, ROC: {roc}, PRAUC: {prauc}, F1: {f1}")

model_save = f"models/{model_name}_{train_ts}.sd"
torch.save(model.state_dict(), model_save)
