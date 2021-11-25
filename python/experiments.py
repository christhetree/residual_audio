import logging
import math
import os
from typing import List

import pytorch_lightning as pl
import soundfile as sf
import torch as tr
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import Tensor as T
from torch.utils.data import DataLoader, random_split
from torchaudio.transforms import GriffinLim
from tqdm import tqdm

from audio_ae import SpecAE
from config import PROC_AUDIO_DIR, BATCH_SIZE, NUM_WORKERS, OUT_DIR, GPU, \
    GPU_BATCH_SIZE, EPS, SR, N_FFT, HOP_LENGTH
from data_preparation import normalize_waveform
from datasets import PathsDataset

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

# tr.manual_seed(42)  # Set for reproducible results


def train(experiment_name: str, n_filters: int) -> None:
    ae = SpecAE(n_filters=n_filters)
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
                       min_delta=0.001,
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
def calc_diff_spec(spec: T, rec: T, alpha: float = 1.0) -> T:
    spec_max = tr.amax(spec, dim=[1, 2], keepdim=True)
    spec = spec / spec_max

    rec_max = tr.amax(rec, dim=[1, 2], keepdim=True)
    rec = rec / rec_max

    diff = tr.clamp(spec - (alpha * rec), 0.0)
    diff *= tr.maximum(spec_max, rec_max)

    return diff


def eval(model_name: str, alphas: List[float]) -> None:
    batch_size = GPU_BATCH_SIZE
    ae = SpecAE.load_from_checkpoint(os.path.join(OUT_DIR, model_name),
                                     batch_size=batch_size)

    # I evaluate locally with different test files
    paths = list(os.listdir(PROC_AUDIO_DIR))
    paths = [os.path.join(PROC_AUDIO_DIR, f)
             for f in paths if f.endswith('.pt')]
    dataset = PathsDataset(paths)
    eval_dl = DataLoader(dataset,
                         batch_size=1,
                         shuffle=False,  # Set to false for reproducible results
                         drop_last=True)
    gl = GriffinLim(n_fft=N_FFT,
                    hop_length=HOP_LENGTH,
                    power=2.0)

    for idx, batch in tqdm(enumerate(eval_dl)):
        spec_norm, rec_norm = ae.forward(batch)
        spec = unnormalize_spec(spec_norm)
        rec = unnormalize_spec(rec_norm)

        for alpha in alphas:
            diff = calc_diff_spec(spec, rec, alpha)

            diff_gl = gl(diff).squeeze(0)
            diff_gl_norm = normalize_waveform(diff_gl)
            save_name = f'{idx:>02}__{alpha:.3f}__diff_gl_norm.wav'

            sf.write(os.path.join(OUT_DIR, save_name),
                     diff_gl_norm.detach().numpy(),
                     samplerate=SR)


if __name__ == '__main__':
    # n_filters = 1
    # experiment_name = f'ae__conv_2__filters_{n_filters}__stride_2'
    # train(experiment_name, n_filters)

    alphas = [0.0, 0.005, 0.02, 0.08, 0.32, 1.28]
    eval(
        'ae__conv_2__filters_1__stride_2__epoch=00__val_loss=0.179.ckpt',
        alphas,
    )
