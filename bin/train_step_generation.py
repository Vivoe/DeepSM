import os
import time

import numpy as np
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, f1_score
from sklearn.preprocessing import label_binarize

from deepSM import SMDataset
from deepSM import SMGenDataset
from deepSM import StepPlacement
from deepSM import StepGeneration
from deepSM import utils
from deepSM import SMDUtils
import torch
from torch import nn
from torch import optim
import torch.utils.data as datautils

import argparse

os.environ['JNG_KEY'] = 'LVZwIQEcU15ur3cvfbOGD3n75etmEP3A2nFg7n8N'

parser = argparse.ArgumentParser()
parser.add_argument('dataset_name')
parser.add_argument('model_name', type=str)
parser.add_argument('--chunk_size', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--output_dir', type=str, default='.')
parser.add_argument('--n_songs', type=int, default=None)

args = parser.parse_args()


dataset_name = args.dataset_name
batch_size = args.batch_size
model_name = args.model_name
chunk_size = args.chunk_size
n_epochs = args.epochs
output_dir = args.output_dir
n_songs = args.n_songs
train_ts = utils.timestamp()


print("Training on dataset", dataset_name)
print("Batch size:", batch_size)
print("Chunk size:", chunk_size)
print("N epochs:", n_epochs)
print("Timestamp:", train_ts)
print("Output dir:", output_dir)

st_time = time.time()

print("Loading data...")

train_dataset = SMDUtils.get_dataset_from_file(
    dataset_name + '_train', 'gen', n_songs=n_songs, chunk_size=chunk_size)

test_dataset = SMDUtils.get_dataset_from_file(
    dataset_name + '_test', 'gen', n_songs=n_songs, chunk_size=chunk_size)

# Train/test sets are pre-generated.
train_loader = datautils.DataLoader(
    train_dataset,
    num_workers=8,
    batch_size = batch_size,
    shuffle=True,
    pin_memory=True)
print("Train dataset size:", len(train_dataset))

test_loader = datautils.DataLoader(
    test_dataset, num_workers=8,
    batch_size=batch_size, pin_memory=True)

print("Test datset size:", len(test_dataset))



model = StepGeneration.RegularizedRecurrentStepGenerationModel(log=True)
model.cuda()

print("Training...")

model.fit(train_loader, n_epochs, batch_size)

print("Model fitted.")

outputs_list, labels_list = model.predict(test_loader, return_list=True)

outputs = np.concatenate(
        list(map(lambda output: output.reshape((-1, 4, 5)).numpy(),
            outputs_list)),
        axis=0)

print(labels_list[0].shape)
labels = np.concatenate(labels_list, axis=0).reshape((-1, 4))

preds = np.argmax(outputs, axis=2)
print(labels.shape)
print(preds.shape)

accuracy = {}
overall_acc = np.mean(labels == preds)
for i in range(5):
    acc = np.mean(labels[labels == i] == preds[labels == i])
    accuracy[i] = acc if not np.isnan(acc) else None


accuracy['overall'] = overall_acc
print("Accuracy:", accuracy)

utils.notify(accuracy)

model_save = f"{output_dir}/models/{model_name}_{train_ts}.sd"

torch.save(model.state_dict(), model_save)

total_time = time.time() - st_time

utils.notify(f"Done. Total time: {utils.format_time(total_time)}")
