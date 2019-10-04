
import os
import shutil

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler

from deepSM import SMData
from deepSM import utils
import deepSM.beat_time_converter as BTC
from deepSM import wavutils
from deepSM import StepPlacement
from deepSM import SMDataset
from deepSM import SMGenDataset

import h5py

from importlib import reload
reload(BTC)
reload(wavutils)
reload(utils)
reload(SMData)
reload(SMDataset)

def load_placement_dataset(smd_dir, **kwargs):
    smds = []
    
    files = os.listdir(smd_dir)
    for fname in files:
        smds.append(SMDataset.load(file))
    
    return ConcatDataset(smds, **kwargs)

# Uncomment when we get to gen datasets.
# def load_placement_dataset(smd_dir, **kwargs):
#     smds = []
    
#     files = os.listdir(smd_dir)
#     for fname in files:
#         smds.append(SMGenDataset.load(file))
    
#     return ConcatDataset(smds, **kwargs)