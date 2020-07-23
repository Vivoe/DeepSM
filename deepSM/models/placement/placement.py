import torch
import torch.nn.functional as F

from pytorch_lightning.logging.neptune import NeptuneLogger
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning.metrics.functional as fmetrics 

from neptunecontrib.api import log_chart

import matplotlib.pyplot as plt

from deepSM import utils

config = utils.config

class PlacementModel(LightningModule):
    """
    Handles PlacementModel testing functionality.
    """
    
    def __init__(self, debug=False):
        super().__init__()

        debug_str = '-debug' if debug else ''

        self.nep_logger = NeptuneLogger(
            api_key=utils.get_neptune_api_token(),
            project_name='vivoe/deepSM-step-placement' + debug_str,
            params=utils.format_neptune_params()
        )

    def test_epoch_end(self, outputs):
        plt.rcParams['figure.figsize'] = (12, 12)

        # Put test metrics here.
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        scores = torch.cat([x['scores'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])

        probs = F.sigmoid(scores)

        step_threshold = config['network']['placement']['stepThreshold']
        preds = probs > step_threshold

        self.nep_logger.experiment.log_metric("percent_positive", preds.float().mean())

        acc = fmetrics.accuracy(preds, labels) 
        self.nep_logger.experiment.log_metric('test_accuracy', acc)

        fpr, tpr, threshs = fmetrics.roc(probs, labels)
        auroc = fmetrics.auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.set_title("ROC")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.plot(fpr.cpu().numpy(), tpr.cpu().numpy())

        self.nep_logger.experiment.log_metric('test_auroc', auroc)
        self.nep_logger.experiment.log_image('roc_figure', fig)

        plt.clf()

        f1 = fmetrics.f1_score(probs, labels)
        self.nep_logger.experiment.log_metric('test_f1', f1)

        precision, recall, threshs = fmetrics.precision_recall_curve(probs, labels)
        prauc = fmetrics.auc(recall, precision)

        fig, ax = plt.subplots()
        ax.set_title("PR Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.plot(recall.cpu().numpy(), precision.cpu().numpy())

        self.nep_logger.experiment.log_metric('test_prauc', prauc)
        self.nep_logger.experiment.log_image('pr_figure', fig)

        plt.clf()
