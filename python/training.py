import logging
import os
import random
from typing import Optional

import pytorch_lightning as pl
import torch as tr
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

from config import BATCH_SIZE, NUM_WORKERS, OUT_DIR, AUDIO_CHUNKS_PT_DIR, \
    N_FFT, HOP_LEN, GPU
from datasets import PathsDataset
from modeling import SpecCNN2D
from pl_wrapper import PLWrapper
from realtime_stft import RealtimeSTFT

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

tr.backends.cudnn.benchmark = True
log.info(f'tr.backends.cudnn.benchmark = {tr.backends.cudnn.benchmark}')
# tr.manual_seed(42)  # Set for reproducible results


def train(experiment_name: str,
          model: LightningModule,
          data_dir: str = AUDIO_CHUNKS_PT_DIR,
          max_N: Optional[int] = None,
          val_split: float = 0.2) -> None:
    paths = list(os.listdir(data_dir))
    paths = [os.path.join(data_dir, f) for f in paths if f.endswith('.pt')]
    random.shuffle(paths)
    if max_N is not None:
        paths = paths[:max_N]

    n = len(paths)
    val_n = int(val_split * n)
    train_n = n - val_n
    train_paths = paths[:train_n]
    val_paths = paths[train_n:]

    log.info(f'dataset n = {n}')
    log.info(f'train_n n = {train_n}')
    log.info(f'val_n n = {val_n}')

    train_ds = PathsDataset(train_paths)
    val_ds = PathsDataset(val_paths)
    train_dl = DataLoader(train_ds,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          drop_last=True,
                          num_workers=NUM_WORKERS)
    val_dl = DataLoader(val_ds,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        drop_last=True,
                        num_workers=NUM_WORKERS)

    cp = ModelCheckpoint(
        dirpath=OUT_DIR,
        filename=experiment_name + '__{epoch:>02}__{val_loss:.3f}',
        auto_insert_metric_name=True,
        monitor='val_loss',
        mode='min',
        save_top_k=2,
        verbose=True
    )
    es = EarlyStopping(monitor='val_loss',
                       mode='min',
                       min_delta=0.001,
                       patience=8,
                       verbose=True)
    trainer = pl.Trainer(gpus=GPU,
                         max_epochs=100,
                         log_every_n_steps=50,
                         callbacks=[cp, es])
    log.info('')
    log.info(f'====================== {experiment_name} ======================')
    log.info('')
    trainer.fit(model, train_dl, val_dl)


if __name__ == '__main__':
    experiment_name = 'testing'
    rts = RealtimeSTFT(batch_size=BATCH_SIZE, n_fft=N_FFT, hop_len=HOP_LEN)
    # ae = SpecMLP()
    # ae = SpecCNN1D()
    ae = SpecCNN2D()
    model = PLWrapper(ae, rts)
    train(experiment_name, model)