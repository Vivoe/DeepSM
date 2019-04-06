import os
import argparse

import numpy as np
from sklearn import metrics, preprocessing
from scipy import special

import torch
import torch.utils.data as datautils

from deepSM import StepGeneration
from deepSM import SMDUtils
from deepSM import utils


parser = argparse.ArgumentParser()
parser.add_argument("gen_model", type=str)
parser.add_argument("dataset_name", type=str)
parser.add_argument('--n_batches', type=int, default=2000)
parser.add_argument('--chunk_size', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=128)

args = parser.parse_args()

print("Testing model", args.gen_model)
print("Datset name:", args.dataset_name)


test_dataset = SMDUtils.get_dataset_from_file(
        args.dataset_name + '_test',
        'gen',
        chunk_size=args.chunk_size)

test_loader = datautils.DataLoader(
        test_dataset,
        num_workers = 4,
        batch_size = args.batch_size)

model = StepGeneration.RegularizedRecurrentStepGenerationModel()
model.load_state_dict(torch.load(args.gen_model))
model.cuda()

# Outputs is of shape [n_points, 4, 5]
outputs, labels = model.predict(test_loader, max_batches = args.n_batches,
        return_list=True)

outputs = torch.cat(outputs).numpy()
labels = torch.cat(labels).numpy()

outputs = outputs.reshape((-1, 4, 5))

scores = special.softmax(outputs, axis=2)
preds = scores.argmax(axis=2)
acc = metrics.accuracy_score(labels.reshape(-1), preds.reshape(-1))
print(f"Accuracy: {acc:.3f}")


# ROC: Per class type.
for step_type in range(5):
    step_scores = scores[:, :, step_type]
    roc = metrics.roc_auc_score(
            labels.reshape(-1) == step_type,
            step_scores.reshape(-1))
    print(f"ROC-AUC for step type {step_type}: {roc:.3f}")

# PR-AUC: Per class type.
for step_type in range(5):
    step_scores = scores[:, :, step_type]
    precision, recall, thresh = metrics.precision_recall_curve(
            labels.reshape(-1) == step_type,
            step_scores.reshape(-1))
    prauc = metrics.auc(recall, precision)
    print(f"PR-AUC for step type {step_type}: {prauc:.3f}")

# F1 score.
for step_type in range(5):
    step_scores = scores[:, :, step_type]
    f1 = metrics.f1_score(
            labels.reshape(-1) == step_type,
            preds.reshape(-1) == step_type)
    print(f"F1 score for step type {step_type}: {f1:.3f}")

# Perplexity: Per step.
entropy = np.sum(scores * np.log2(scores + 1e-8), axis=2)
perplexity = np.mean(2**-entropy)
print(f"Average perplexity: {perplexity:.3f}")

# Blank notes
blanks = (preds.sum(axis=1) == 0).mean()
print(f"Percent blank: {blanks:.3f}")

# Triple or more
triples = (preds.sum(axis=1) >= 3).mean()
print(f"Percent triple+: {blanks:.3f}")

utils.notify('DONE')

