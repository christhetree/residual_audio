import logging
import os
from typing import Union

import librosa as lr
import soundfile as sf
import torch as tr
from joblib import Parallel, delayed
from numpy import ndarray
from torch import Tensor as T
from tqdm import tqdm

from config import SR, N_SAMPLES, PEAK_VALUE, RAW_AUDIO_DIR, PROC_AUDIO_DIR, \
    STEP_SIZE_SECONDS, MAX_N_STEPS

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def normalize_waveform(x: Union[T, ndarray],
                       peak_value: float = PEAK_VALUE) -> Union[T, ndarray]:
    assert len(x.shape) == 1
    return (x / max(abs(x.max()), abs(x.min()))) * peak_value


# Use this for parallel processing over an existing audio snippet dataset
def prepare_short_audio_parallel(in_dir: str,
                                 out_dir: str = PROC_AUDIO_DIR) -> None:
    names = list(os.listdir(in_dir))
    Parallel(n_jobs=-1, verbose=5)(
        delayed(prepare_short_audio)(in_dir, name, out_dir)
        for name in names
    )


# Use this for parallel processing over an existing audio snippet dataset
def prepare_short_audio(in_dir: str,
                        name: str,
                        out_dir: str = PROC_AUDIO_DIR) -> None:
        try:
            audio, _ = lr.load(os.path.join(in_dir, name), sr=SR, mono=True)
        except:
            log.warning(f'Failed to load: {name}')
            return

        audio = audio[:N_SAMPLES]
        if len(audio) != N_SAMPLES:
            log.warning(f'Audio too short: {name}')
            return

        audio = normalize_waveform(audio)
        audio = tr.tensor(audio)
        if tr.isnan(audio).any():
            log.warning('NaN found!')
            return

        tr.save(audio, os.path.join(out_dir, f'{name[:-4]}.pt'))


# Use this for long audio from videos etc.
def prepare_raw_audio(in_dir: str,
                      name: str,
                      out_dir: str = PROC_AUDIO_DIR,
                      step_size_seconds: float = STEP_SIZE_SECONDS,
                      max_n_steps: int = MAX_N_STEPS,
                      save_wav: bool = False) -> None:
        try:
            audio, _ = lr.load(os.path.join(in_dir, name), sr=SR, mono=True)
        except:
            log.warning(f'Failed to load: {name}')
            return

        min_n_samples_per_step = int(step_size_seconds * SR)
        step_n_samples = len(audio) // max_n_steps
        step_n_samples = max(min_n_samples_per_step, step_n_samples)
        n_steps = len(audio) // step_n_samples
        for idx in tqdm(range(n_steps)):
            start = idx * step_n_samples
            end = start + N_SAMPLES
            snippet = audio[start:end]
            if len(snippet) != N_SAMPLES:
                break

            snippet = normalize_waveform(snippet)
            if save_wav:
                sf.write(os.path.join(out_dir, f'{name[:-4]}__{idx:>05}.wav'),
                         snippet,
                         samplerate=SR)

            snippet = tr.tensor(snippet)
            if tr.isnan(snippet).any():
                log.warning('NaN found!')
                continue

            tr.save(snippet,
                    os.path.join(out_dir, f'{name[:-4]}__{idx:>05}.pt'))


if __name__ == '__main__':
    for name in tqdm(os.listdir(RAW_AUDIO_DIR)):
        prepare_raw_audio(
            RAW_AUDIO_DIR,
            name,
            # step_size_seconds=3.0,  # For less overlap in snippets
            # save_wav=True  # For listening to the output
        )
