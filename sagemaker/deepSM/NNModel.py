import torch
from torch import nn
from torch import optim

import numpy as np
import time

from deepSM import utils


class NNModel(nn.Module):
    def __init__(self):
        self.use_cuda = False
        super().__init__()

    def forward(self, x, diff):
        raise NotImplementedError

    def compute_loss(self, criterion, outputs, labels):
        raise NotImplementedError

    def get_criterion(self):
        raise NotImplementedError

    def prepare_data(self, batch):
        raise NotImplementedError

    def get_optim(self):
        return optim.Adam(self.parameters())

    def fit(self, data_loader, n_epochs, batch_size, checkpoint_freq=None, checkpoint_dir=None):

        criterion = self.get_criterion()
        optimizer = self.get_optim()

        batch_timer = 0
        batch_samples = 50
        loss_progress = []
        st = time.time()
        checkpoint_time = time.time()
        for epoch in range(n_epochs):
            print("Starting epoch", epoch+1)

            for i, batch_data in enumerate(data_loader):
                # Prepare data and take step.
                data, labels = self.prepare_data(batch_data)

                optimizer.zero_grad()

                with torch.autograd.detect_anomaly():
                    outputs = super().__call__(*data)
                    loss = self.compute_loss(criterion, outputs, labels)
                    loss.backward()
                    optimizer.step()

                # Progress tracking.
                if i % 25 == 0:
                    loss_progress.append(loss.item())
                if i % 500 == 499:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch+1, i+1, np.mean(loss_progress[-10:])))
                
                if checkpoint_dir and time.time() - checkpoint_time > checkpoint_freq:
                    torch.save(model.state_dict(), f'{checkpoint_dir}/StepPlacement.torch')

                batch_timer += 1
                if batch_timer == batch_samples:
                    batch_time = (time.time() - st) / batch_samples
                    batches_per_epoch = len(data_loader)
                    expected_time = \
                        batch_time * batches_per_epoch * n_epochs * 0.8

                    print("Run time per batch: %s" %
                          utils.format_time(batch_time))
                    print("Crude expected runtime: %s" %
                          utils.format_time(expected_time))
                    
                    print("TEMPORARY BREAK")
                    break  # Temporary

            if epoch == 0:
                epoch_time = time.time() - st
                remaining_time = epoch_time * (n_epochs - 1)
                epoch_time_str = utils.format_time(epoch_time)
                remaining_time_str = utils.format_time(remaining_time)
                print("Epoch time: %s" % epoch_time_str)
                print("Expected remaining running time: %s" %
                      remaining_time_str)

                utils.notify("First epoch done! \nEpoch time: %s" %
                             epoch_time_str +
                             "\nRemaining time: %s" %
                             remaining_time_str)

        # Done training.
        final_loss = np.mean(loss_progress[-10:])
        print("Training complete.")
        print(f"Final loss: {final_loss:.5f}")
        utils.notify(f"Training done! Final loss: {final_loss:.5f}")


    def predict(self, data_loader, max_batches=None, return_list=False):
        """
        return_list is used when a non-even seq length is used.
        """

        N = len(data_loader.dataset)
        batch_size = data_loader.batch_size
        print(f"Dataset size: {N}")
        print(f"N batches: {N // batch_size + 1}")

        labels_list = []
        outputs_list = []

        self.eval()

        with torch.no_grad():
            for i, batch_data in enumerate(data_loader):
                if max_batches is not None and i >= max_batches:
                    break

                if i % 500 == 0:
                    print("Batch ", i)

                data, labels = self.prepare_data(batch_data)
                outputs = super().__call__(*data)

                labels_list.append(labels.cpu())
                outputs_list.append(outputs.cpu())

        if return_list:
            return outputs_list, labels_list
        else:
            outputs = torch.cat(outputs_list).numpy().reshape(-1)
            labels = torch.cat(labels_list).numpy().reshape(-1)
            return outputs, labels


    def cuda(self, *args):
        self.use_cuda = True
        return super().cuda(*args)
