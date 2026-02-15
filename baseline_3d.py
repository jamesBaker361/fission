

import argparse
import logging
import os
import subprocess
import sys
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from PIL import Image

import threestudio
from threestudio.systems.base import BaseSystem
from threestudio.utils.callbacks import CustomProgressBar
from threestudio.utils.config import ExperimentConfig, load_config
from torch.utils.data import random_split
from threestudio.utils.misc import get_rank
CLUSTER_ROOT = "/umbc/rs/pi_donengel/users/jbaker15/fission"
THREESTUDIO_DIR = os.path.join(CLUSTER_ROOT, "threestudio")
sys.path.insert(0, CLUSTER_ROOT)
#sys.path.insert(0, THREESTUDIO_DIR)
from data_helpers import ShapeNetImageDataPaired

def main():
    dim=(64,64) #change this
    generator=torch.Generator()
    generator.manual_seed(123)

    train_dataset=ShapeNetImageDataPaired("shapenet_renders",dim=-1)
    test_dataset,val_dataset,train_dataset=random_split(train_dataset,[0.1,0.1,0.8],generator=generator)
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)
    
    config_path = os.path.join(THREESTUDIO_DIR, "configs", "dreamfusion.yaml")
    for batch in test_loader:
        image=batch["image"]
        image.save("temp.png")
        processed_image="img.jpg" #change ofc
        cli_overrides = [
            f"data.image_path=temp.png",
        ]
        cfg: ExperimentConfig = load_config(config_path, cli_args=cli_overrides, n_gpus=1)
        pl.seed_everything(cfg.seed + get_rank(), workers=True)
        dm = threestudio.find(cfg.data_type)(cfg.data)
        system: BaseSystem = threestudio.find(cfg.system_type)(cfg.system, resumed=False)
        system.set_save_dir(os.path.join(cfg.trial_dir, "save"))
        
        callbacks = []
        os.makedirs(os.path.join(cfg.trial_dir, "tb_logs"), exist_ok=True)
        loggers = []

        # ── trainer ────────────────────────────────────────────────────────────
        trainer = Trainer(
            callbacks=callbacks,
            logger=loggers,
            inference_mode=False,
            accelerator="gpu",
            devices=-1,
            **cfg.trainer,
        )
        
        trainer.fit(system, datamodule=dm)
        trainer.test(system, datamodule=dm)
        
        break