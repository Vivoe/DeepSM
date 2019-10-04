import os
import time
import re
import argparse
import io

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score

from deepSM import SMDataset
from deepSM import SMDUtils
from deepSM import StepPlacement
from deepSM import utils
from deepSM import samplers

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data as datautils

def train(args):
    print("Begin training.")
    train_dataset = SMDUtils.get_dataset_from_file(args.train)
    test_dataset = SMDUtils.get_dataset_from_file(args.test)
    
    train_loader = datautils.DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True)
    
    test_loader = datautils.DataLoader(
        test_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True)
    
    print('N train:', len(train_loader))
    print('N test:', len(test_loader))
    
    train_ts = utils.timestamp()
    
    model = StepPlacement.RecurrentStepPlacementModel()
    
    checkpoint_path = f'{args.output_data_dir}/model.cpt'
    if os.path.exists(checkpoint_path):
        print("Loading weights from" checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path))
        
    model.cuda()
    
    model.fit(train_loader, 
              args.epochs, 
              args.batch_size, 
              args.checkpoint_freq,
              args.output_data_dir)
    
    torch.save(model.state_dict(), f'{args.model_dir}/StepPlacement.torch')
    
    outputs, labels = model.predict(test_loader, max_batches = 2000)
    
    s3_bucket = 'sagemaker-us-west-1-284801879240'
    sm_env = json.loads(os.environ['SM_TRAINING_ENV'])
    s3_path = sm_env['job_name']
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    preds = sigmoid(outputs) > 0.1
    
    accuracy = 1 - np.mean(np.abs(preds - labels))
    print("Accuracy:", accuracy)
    
    percent_pos = np.mean(preds)
    print("Percent positive:", percent_pos)
    
    fpr, tpr, _ = roc_curve(labels, outputs)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr)
    roc_buf = io.BytesIO()
    plt.savefig(roc_buf, format='png')
    roc_buf.seek(0)
    utils.upload_image_obj(roc_buf, s3_bucket, f'{s3_path}/roc_auc.png')
    
    print("AUC ROC:", roc_auc)
    
    precision, recall, _ = precision_recall_curve(labels, outputs)
    prauc = auc(recall, precision)
    print("AUC PR:", prauc)
    
    plt.plot(recall, precision)
    pr_buf = io.BytesIO()
    plt.savefig(pr_buf, format='png')
    pr_buf.seek(0)
    utils.upload_image_obj(pr_buf, s3_bucket, f'{s3_path}/pr_auc.png')
    
    f1 = f1_score(labels, preds)
    print("F1 score:", f1)
    
    output_metrics = [
        'Training done.',
        f'Accuracy: {accuracy}',
        f'Percent pos: {percent_pos}',
        f'ROC AUC: {roc_auc}',
        f'PRAUC: {prauc}',
        f'F1 score: {f1_score}'
    ]
    
    output_str = output_metrics.join('\\n')
    
    utils.notify(output_str)


    
if __name__ =='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--checkpoint-freq', type=int, default=5, help='Checkpoint frequency in minutes.')

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args, _ = parser.parse_known_args()