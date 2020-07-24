import argparse
import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from deepSM.models.placement import conv_placement
from deepSM import utils

parser = argparse.ArgumentParser()
parser.add_argument('--max_epochs', type=int, default=10)
parser.add_argument('--tags', type=str)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

print(args)

tags = [] if args.tags is None else args.tags.split(',')

if args.debug:
    base_path = '/home/lence/dev/DeepSM/data/development/ddrA'
    data_path = f'{base_path}/train/Hopeful/'
    val_path = f'{base_path}/validation/Determination/'
    test_path = f'{base_path}/validation/Rejoin/'
    extra_args = {
        'val_path': val_path,
        'test_path': test_path
    }
else:
    data_path = '/home/lence/dev/DeepSM/data/development/ddrA/'
    extra_args = {}

model = conv_placement.ConvPlacementModel(
        data_path, tags=tags, debug=args.debug,
        **extra_args)

exp_name = model.nep_logger._experiment_id

ckpt_path = f"{utils.BASE_PATH}/checkpoints/{exp_name}/"
os.mkdir(ckpt_path)
print("Checkpoint path: ", ckpt_path) 
 
ckpt_callback = ModelCheckpoint(
    ckpt_path, 
    save_last=True, 
    save_top_k=2,
    period=0.5)

trainer = Trainer(
    gpus=1,
    max_epochs=args.max_epochs,
    logger = model.nep_logger,
    checkpoint_callback=ckpt_callback,
    early_stop_callback=True,
    limit_val_batches=0.5,
    val_check_interval=0.5
)

trainer.fit(model)

model.nep_logger.experiment.log_artifact(ckpt_callback.best_model_path)
model.nep_logger.experiment.log_artifact(f"{ckpt_path}/last.ckpt")

trainer.test()
model.nep_logger.experiment.stop()
