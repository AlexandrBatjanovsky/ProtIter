#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'


import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from deephd_data import DHDDataset
from deephd_model import ASPPResNetSE
from deephd_losses import build_loss_by_name
from deephd_pipeline import DeepHDPipeline
#
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger



def main_train():
    logging.basicConfig(level=logging.INFO)
    # path_cfg = '/mnt/data4t3/data/deepdocking_experiments/homodimers/raw/cfg.json'
    path_cfg = '/mnt/data2t2/data/annaha/cfg_21_02_cb_cont8_tr6.json'
    pipeline = DeepHDPipeline(path_cfg, num_workers=2).build()
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(pipeline.path_model, 'results'),
                                          verbose=True, monitor='val_loss', mode='min')
    logger = TestTubeLogger(save_dir=pipeline.path_model, version=1)
    t1 = time.time()
    trainer = Trainer(default_save_path=pipeline.path_model,
                      logger=logger,
                      log_gpu_memory=False,
                      max_nb_epochs=pipeline.cfg['epochs'],
                      checkpoint_callback=checkpoint_callback,
                      early_stop_callback=False,
                      gpus=[0])
    #distributed_backend='ddp')
    trainer.fit(pipeline)
    dt = time.time() - t1
    logging.info(f'\t\t... done, dt ~ {dt:0.2f} (s)')


if __name__ == '__main__':
    main_train()