import logging
import math
import os
import random

import pytorch_lightning as pl
import soundfile as sf
import torch as tr
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import Tensor as T
from torch.utils.data import DataLoader, random_split
from torchaudio.transforms import GriffinLim

from audio_ae import SpecAE
from config import PROC_AUDIO_DIR, BATCH_SIZE, NUM_WORKERS, OUT_DIR, GPU, \
    GPU_BATCH_SIZE, EPS, SR, N_FFT, HOP_LENGTH
from data_preparation import normalize_waveform
from datasets import PathsDataset

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

# tr.manual_seed(42)  # Set for reproducible results


def train(experiment_name: str) -> None:
    ae = SpecAE()
    val_split = 0.2

    paths = list(os.listdir(PROC_AUDIO_DIR))
    paths = [os.path.join(PROC_AUDIO_DIR, f)
             for f in paths if f.endswith('.pt')]
    dataset = PathsDataset(paths)

    n = len(dataset)
    val_n = int(val_split * n)
    train_n = n - val_n
    train_ds, val_ds, = random_split(dataset, [train_n, val_n])

    log.info(f'dataset n = {n}')
    log.info(f'train_n n = {train_n}')
    log.info(f'val_n n = {val_n}')

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
    # tr.save(train_dl, os.path.join(OUT_DIR, 'train_dl.pt'))
    # tr.save(val_dl, os.path.join(OUT_DIR, 'val_dl.pt'))

    cp = ModelCheckpoint(
        dirpath=OUT_DIR,
        filename=experiment_name + '__{epoch:>02}__{val_loss:.3f}',
        auto_insert_metric_name=True,
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        verbose=True
    )
    es = EarlyStopping(monitor='val_loss',
                       mode='min',
                       min_delta=0.0005,
                       patience=8,
                       verbose=True)

    trainer = pl.Trainer(gpus=GPU,
                         progress_bar_refresh_rate=1,
                         max_epochs=10,
                         log_every_n_steps=100,
                         callbacks=[cp, es])
    log.info('')
    log.info(f'====================== {experiment_name} ======================')
    log.info('')
    trainer.fit(ae, train_dl, val_dl)


def unnormalize_spec(spec_norm: T, eps: float = EPS) -> T:
    spec = (10 ** (spec_norm * (-math.log10(eps)))) - eps
    spec = tr.clamp(spec, 0.0)
    return spec


# TODO(christhetree): experiment with different difference calculations
def calc_diff_spec(spec: T, rec: T) -> T:
    spec_max = tr.amax(spec, dim=[1, 2], keepdim=True)
    spec = spec / spec_max

    rec_max = tr.amax(rec, dim=[1, 2], keepdim=True)
    rec = rec / rec_max

    diff = tr.clamp(spec - rec, 0.0)
    diff *= tr.maximum(spec_max, rec_max)

    return diff


def eval(model_name: str, n_samples: int = 4) -> None:
    batch_size = GPU_BATCH_SIZE
    ae = SpecAE.load_from_checkpoint(os.path.join(OUT_DIR, model_name),
                                     batch_size=batch_size)

    # I evaluate locally with different test files
    paths = list(os.listdir(PROC_AUDIO_DIR))
    paths = [os.path.join(PROC_AUDIO_DIR, f)
             for f in paths if f.endswith('.pt')]
    dataset = PathsDataset(paths)
    eval_dl = DataLoader(dataset,
                         batch_size=batch_size,
                         shuffle=False,  # Set to false for reproducible results
                         drop_last=True)
    batch = next(iter(eval_dl))

    log.info(f'Performing inference on batch')
    spec_norm, rec_norm = ae.forward(batch)
    spec = unnormalize_spec(spec_norm)
    rec = unnormalize_spec(rec_norm)
    diff = calc_diff_spec(spec, rec)

    gl = GriffinLim(n_fft=N_FFT,
                    hop_length=HOP_LENGTH,
                    power=2.0)

    # for idx in [0, 1]:
    for idx in random.sample(range(len(batch)), n_samples):
        log.info(f'Sample batch idx: {idx}')
        sf.write(os.path.join(OUT_DIR, f'{model_name}_{idx:>02}_orig.wav'),
                 batch[idx].detach().numpy(),
                 samplerate=SR)

        # GL operations are expensive
        # spec_gl = gl(spec[idx].unsqueeze(0))
        rec_gl = gl(rec[idx].unsqueeze(0)).squeeze(0)
        diff_gl = gl(diff[idx].unsqueeze(0)).squeeze(0)
        diff_gl_norm = normalize_waveform(diff_gl)

        sf.write(os.path.join(OUT_DIR, f'{model_name}_{idx:>02}_rec_gl.wav'),
                 rec_gl.detach().numpy(),
                 samplerate=SR)
        sf.write(os.path.join(OUT_DIR, f'{model_name}_{idx:>02}_diff_gl_norm.wav'),
                 diff_gl_norm.detach().numpy(),
                 samplerate=SR)


if __name__ == '__main__':
    experiment_name = 'ae__2conv1d_32filters_2stride'
    train(experiment_name)

    # eval('testing__epoch=00__val_loss=0.209.ckpt')

    # eval('ae__1conv1d_16filters_2stride__epoch=00__val_loss=0.077.ckpt')
    # eval('ae__1conv1d_128filters_2stride__epoch=00__val_loss=0.072.ckpt')
    # eval('ae__1conv1d_512filters_2stride__epoch=00__val_loss=0.066.ckpt')
    # eval('ae__1conv1d_512filters_2stride__epoch=06__val_loss=0.048.ckpt')
    # eval('ae__1conv1d_512filters_2stride__epoch=06__val_loss=0.048.ckpt')

    # eval('ae__2conv_16filters_2stride__epoch=00__val_loss=0.049.ckpt')

    # eval('ae__2conv_1filters__epoch=00__val_loss=0.111.ckpt')
    # eval('ae__2conv_4filters__epoch=00__val_loss=0.092.ckpt')
    # eval('ae__2conv_16filters__epoch=00__val_loss=0.061.ckpt')
    # eval('ae__2conv_32filters__epoch=00__val_loss=0.053.ckpt')
    # eval('ae__2conv_64filters__epoch=00__val_loss=0.046.ckpt')
