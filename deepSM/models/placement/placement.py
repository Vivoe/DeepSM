import torch
import torch.nn.functional as F

from pytorch_lightning.logging.neptune import NeptuneLogger
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning.metrics.functional as fmetrics 

import matplotlib.pyplot as plt

from deepSM import utils

config = utils.config

class PlacementModel(LightningModule):
    """
    Handles PlacementModel testing functionality.
    """
    
    def __init__(self, 
        learning_rate=config['network']['placement']['lr'], 
        debug=False, 
        tags=[],
        **kwargs):

        super().__init__()

        self.learning_rate = learning_rate

        debug_str = '-debug' if debug else ''

        self.nep_logger = NeptuneLogger(
            api_key=utils.get_neptune_api_token(),
            project_name='vivoe/deepSM-step-placement' + debug_str,
            params=utils.format_neptune_params(),
            tags=['placement-model'] + tags
        )


    def log_metric(self, name, metric):
        self.nep_logger.experiment.log_metric(name, metric)


    def validation_epoch_end(self, outputs):
        scores = torch.cat([x['scores'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])

        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        probs = torch.sigmoid(scores)
        step_threshold = config['network']['placement']['stepThreshold']
        preds = probs > step_threshold

        self.log_metric("val_avg_loss", avg_loss)
        self.log_metric("val_percent_positive", preds.float().mean())

        try:
            acc = fmetrics.accuracy(preds, labels) 
        except RuntimeError:
                acc = 0
        self.log_metric("val_acc", acc)

        try:
            fpr, tpr, threshs = fmetrics.roc(scores, labels)
            auroc = fmetrics.auc(fpr, tpr)
        except ValueError:
            auroc = 0.5
        self.log_metric("val_auroc", auroc)

        precision, recall, threshs = fmetrics.precision_recall_curve(scores, labels)
        prauc = fmetrics.auc(recall, precision)
        if torch.isnan(prauc):
            prauc = 0
        self.log_metric("val_prauc", prauc)

        return {'val_loss': prauc}


    def test_epoch_end(self, outputs):


        plt.rcParams['figure.figsize'] = (12, 12)

        # Put test metrics here.
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.log_metric("test_avg_loss", avg_loss)

        scores = torch.cat([x['scores'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])

        # Prediction-only metrics
        probs = F.sigmoid(scores)
        step_threshold = config['network']['placement']['stepThreshold']
        preds = probs > step_threshold
        self.nep_logger.experiment.log_metric("percent_positive", preds.float().mean())

        fig, ax = plt.subplots()
        ax.set_title('pred histogram')
        ax.hist(probs.cpu().numpy(), bins=50)
        self.nep_logger.experiment.log_image('pred_histogram', fig)
        plt.clf()


        acc = fmetrics.accuracy(preds, labels) 
        self.nep_logger.experiment.log_metric('test_accuracy', acc)

        fpr, tpr, threshs = fmetrics.roc(scores, labels)
        auroc = fmetrics.auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.set_title("ROC")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.plot(fpr.cpu().numpy(), tpr.cpu().numpy())

        self.nep_logger.experiment.log_metric('test_auroc', auroc)
        self.nep_logger.experiment.log_image('roc_figure', fig)

        plt.clf()

        f1score = fmetrics.f1_score(scores, labels)
        self.nep_logger.experiment.log_metric('test_f1', f1score)

        precision, recall, threshs = fmetrics.precision_recall_curve(scores, labels)
        prauc = fmetrics.auc(recall, precision)

        fig, ax = plt.subplots()
        ax.set_title("PR Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.plot(recall.cpu().numpy(), precision.cpu().numpy())

        self.nep_logger.experiment.log_metric('test_prauc', prauc)
        self.nep_logger.experiment.log_image('pr_figure', fig)

        plt.clf()