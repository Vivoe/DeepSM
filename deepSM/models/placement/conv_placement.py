import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule

from deepSM import utils
from deepSM.data import smdataset
import deepSM.models.placement.placement as pl

from importlib import reload
reload(smdataset)
reload(pl)
config = utils.config

class ConvPlacementModel(pl.PlacementModel):

    def __init__(self, datapath, debug=False, **kwargs):
        """
        Datapath should point to a folder with train/test folders.
        """
        super().__init__(debug=debug, **kwargs)
        self.datapath = datapath
        self.debug = debug
        self.kwargs = kwargs


        configA = config['network']['conv']['convA']
        self.convA = ConvPath(configA['width'], configA['height'])

        configB = config['network']['conv']['convB']
        self.convB = ConvPath(configB['width'], configB['height'])

        configFC = config['network']['placement']['fcLayers']

        fcs = []
        fcs.append(nn.Linear(
            self.convA.output_size + self.convB.output_size + 1,
            configFC[0]
        ))
        
        for i in range(1, len(configFC)):
            fcs.append(nn.Linear(
                configFC[i-1],
                configFC[i]
            ))

        fcs.append(nn.Linear(
            configFC[-1],
            1
        ))

        self.fcs = nn.ModuleList(fcs)
    

    def forward(self, inputs):
        fft_features, diff = inputs

        xa = self.convA(fft_features)
        xb = self.convB(fft_features)
        
        x = torch.cat([
            torch.flatten(xa, start_dim=1), 
            torch.flatten(xb, start_dim=1), 
            diff.float().reshape((-1, 1))
        ], dim=1)

        for fc in self.fcs[:-1]:
            x = F.relu(fc(x))

        return self.fcs[-1](x)


    def training_step(self, batch, batch_idx):
        fft_features = batch['fft_features']
        diff = batch['diff']
        timing_labels = batch['timing_labels']

        y_hat = self((fft_features, diff)).reshape(-1)
        loss = F.binary_cross_entropy_with_logits(y_hat, timing_labels)
        log = {
            'train_loss': loss,
            'y_hat_mean': y_hat.mean()
        }
        return {'loss': loss, 'log': log}


    def validation_step(self, batch, batch_idx):
        fft_features = batch['fft_features']
        diff = batch['diff']
        timing_labels = batch['timing_labels']

        y_hat = self((fft_features, diff)).reshape(-1)
        loss = F.binary_cross_entropy_with_logits(y_hat, timing_labels)
        return {
            'test_loss': loss, 
            'scores': y_hat, 
            'labels': timing_labels,
        }


    def test_step(self, batch, batch_idx):
        fft_features = batch['fft_features']
        diff = batch['diff']
        timing_labels = batch['timing_labels']
        fuzzy_labels = batch['fuzzy_timing_labels']

        y_hat = self((fft_features, diff)).reshape(-1)
        loss = F.binary_cross_entropy_with_logits(y_hat, timing_labels)
        return {
            'test_loss': loss, 
            'scores': y_hat, 
            'labels': timing_labels,
            'fuzzy_labels': fuzzy_labels
        }


    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate
            # lr=config['network']['placement']['lr']
        )


    def train_dataloader(self):
        if self.debug:
            # Intended to have a single song in the file path.
            dataset = smdataset.SMDataset(self.datapath)
        else:
            dataset = smdataset.getSMDataset(self.datapath + '/train') 
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config['network']['placement']['batchSize'],
            shuffle=True,
            num_workers=4
        ) 
        return dataloader


    def val_dataloader(self):
        if self.debug:
            # Intended to have a single song in the file path.
            if 'val_path' in self.kwargs:
                dataset = smdataset.SMDataset(self.kwargs['val_path'])
            else:
                dataset = smdataset.SMDataset(self.datapath)
        else:
            dataset = smdataset.getSMDataset(self.datapath + '/validation') 
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config['network']['placement']['batchSize'],
            num_workers=4
        ) 
        return dataloader


    def test_dataloader(self):
        if self.debug:
            # Intended to have a single song in the file path.
            if 'test_path' in self.kwargs:
                dataset = smdataset.SMDataset(self.kwargs['test_path'])
            else:
                dataset = smdataset.SMDataset(self.datapath)
        else:
            dataset = smdataset.getSMDataset(self.datapath + '/test') 
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config['network']['placement']['batchSize'],
            num_workers=4
        ) 

        return dataloader


         

class ConvPath(nn.Module):
    def __init__(self, band_width, frame_length):
        super().__init__()

        # Dimensions of the leading conv kernels.
        self.band_width = band_width
        self.frame_length = frame_length

        # Dimensions of the input spectrogram slices.
        slice_width = config['audio']['contextWindowSize'] * 2 + 1
        slice_height = config['audio']['fft']['nMels']
        input_channels = len(config['audio']['fft']['nfft'])
        conv_conf = config['network']['conv']
        output_width = conv_conf['convOutput']['width']
        output_height = conv_conf['convOutput']['height']

        self.conv_channels = config['network']['conv']['convChannels']

        self.pool_output_sizes = [
            (slice_width // 2**(i+1), slice_height // 2**(i+1))
            for i in range(len(self.conv_channels))
        ]

        self.output_size = (
            self.pool_output_sizes[-1][0] * 
            self.pool_output_sizes[-1][1] * 
            self.conv_channels[-1]
        )

        conv_modules = []
        bn_modules = []

        conv_modules.append(nn.Conv2d(
            input_channels,
            self.conv_channels[0],
            (band_width, frame_length)
        ))
        bn_modules.append(nn.BatchNorm2d(self.conv_channels[0]))

        for i in range(1, len(self.conv_channels)):
            conv_modules.append(nn.Conv2d(
                self.conv_channels[i-1],
                self.conv_channels[i],
                (output_width, output_height)
            ))

            bn_modules.append(nn.BatchNorm2d(self.conv_channels[i]))

        self.conv_modules = nn.ModuleList(conv_modules)
        self.bn_modules = nn.ModuleList(bn_modules)

    
    def forward(self, x):
        for pool_output_size, bn, conv in zip(self.pool_output_sizes, self.bn_modules, self.conv_modules):
            x = F.relu(bn(conv(x)))
            x = F.adaptive_max_pool2d(x, pool_output_size)

        return x