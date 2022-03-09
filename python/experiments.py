import logging
import math
import os
import random
from typing import List

import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import torch as tr
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import Tensor as T
from torch.utils.data import DataLoader, random_split
from torchaudio.transforms import Fade
from tqdm import tqdm

from audio_ae import SpecAE, ResidualAudioEffect
from config import PROC_AUDIO_DIR, BATCH_SIZE, NUM_WORKERS, OUT_DIR, GPU, \
    EPS, SR, N_SAMPLES
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
                         max_epochs=1,
                         log_every_n_steps=50,
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
    batch_size = BATCH_SIZE
    ae = SpecAE.load_from_checkpoint(os.path.join(OUT_DIR, model_name),
                                     batch_size=batch_size)
    ra_effect = ResidualAudioEffect(ae)
    # script = ra_effect.to_torchscript()
    # tr.jit.save(script, os.path.join(OUT_DIR, 'script_2048_samples.pt'))
    # exit()

    # I evaluate locally with different test files
    paths = list(os.listdir(PROC_AUDIO_DIR))
    paths = [os.path.join(PROC_AUDIO_DIR, f)
             for f in paths if f.endswith('.pt')]
    random.shuffle(paths)
    dataset = PathsDataset(paths)
    eval_dl = DataLoader(dataset,
                         batch_size=1,
                         shuffle=False,  # Set to false for reproducible results
                         drop_last=True)
    overlap_samples = N_SAMPLES // 4
    fade = Fade(overlap_samples, overlap_samples)

    for idx, (path, batch) in tqdm(enumerate(zip(paths, eval_dl))):
        file_name = os.path.basename(path)
        for alpha in alphas:
            buffers = []
            batch = batch.squeeze(0)
            n_steps = len(batch) // (N_SAMPLES - overlap_samples)
            for buffer_idx in range(0, n_steps - 1):
                start_idx = buffer_idx * (N_SAMPLES - overlap_samples)
                end_idx = start_idx + N_SAMPLES
                audio = batch[start_idx:end_idx]
                diff_gl = ra_effect(audio, tr.tensor(alpha))
                buffers.append(diff_gl.detach())

            buffers = [fade(b).numpy() for b in buffers]
            new_buffers = []
            for idx, b in enumerate(buffers[:-1]):
                next_b = buffers[idx + 1]
                b[-overlap_samples:] += next_b[:overlap_samples]
                new_b = b[overlap_samples:]
                new_buffers.append(new_b)

            save_name = f'{file_name}__{alpha:.3f}__diff_gl.wav'
            out_audio = np.concatenate(new_buffers)
            sf.write(os.path.join(OUT_DIR, save_name),
                     out_audio,
                     samplerate=SR)


if __name__ == '__main__':
    # n_filters = 1
    # experiment_name = f'ae__samples_{N_SAMPLES}__conv_2__filters_{n_filters}__stride_2'
    # train(experiment_name, n_filters)
    # exit()

    # alphas = [0.0, 0.005, 0.02, 0.08, 0.32, 1.28]
    alphas = [0.0, 0.02, 0.1, 0.5, 1.0]
    eval(
        'ae__samples_2048__conv_2__filters_1__stride_2__epoch=00__val_loss=0.197.ckpt',
        alphas,
    )
