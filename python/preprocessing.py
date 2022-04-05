import logging
import os
from typing import Optional

import joblib
import librosa as lr
import numpy as np
import torch as tr
from pedalboard import Pedalboard
from tqdm import tqdm

from config import AUDIO_CHUNKS_PT_DIR, SR, RAW_AUDIO_DIR, \
    N_FFT, HOP_LEN, MODEL_IO_N_FRAMES
from realtime_stft import RealtimeSTFT

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def create_chunks(path: str,
                  rts: RealtimeSTFT,
                  save_dir: str,
                  fx: Optional[Pedalboard] = None,
                  fx_save_dir: Optional[str] = None,
                  overlap: float = 0.5,
                  sr: int = SR) -> None:
    try:
        dry_audio, _ = lr.load(path, sr=sr, mono=True)
    except:
        log.warning(f'Failed to load: {path}')
        return

    wet_audio = None
    if fx is not None:
        wet_audio = fx(np.expand_dims(dry_audio, 0), sr).squeeze()
        assert len(dry_audio) == len(wet_audio)
    if fx_save_dir is not None:
        assert fx is not None

    audio_len = len(dry_audio)
    proc_window_len = (rts.model_io_n_frames - 1) * rts.hop_len
    if not rts.center:
        proc_window_len += 2 * rts.overlap_n_frames * rts.hop_len
    proc_hop_len = int(proc_window_len * overlap)
    n_steps = ((audio_len - proc_window_len) // proc_hop_len) + 1
    file_name = os.path.basename(path)

    for idx in range(n_steps):
        start_idx = idx * proc_hop_len
        end_idx = start_idx + proc_window_len
        audio_chunk = dry_audio[start_idx:end_idx]
        assert len(audio_chunk) == proc_window_len
        audio_pt = tr.tensor(audio_chunk)
        save_name = f'{file_name}__{idx:04}.pt'
        save_path = os.path.join(save_dir, save_name)
        tr.save(audio_pt, save_path)

        if fx_save_dir is not None:
            fx_audio_chunk = wet_audio[start_idx:end_idx]
            fx_save_path = os.path.join(fx_save_dir, save_name)
            fx_audio_pt = tr.tensor(fx_audio_chunk)
            tr.save(fx_audio_pt, fx_save_path)

        # spec_chunk = rts.audio_to_spec_offline(audio_pt.unsqueeze(0))
        # import matplotlib.pyplot as plt
        # plt.imshow(spec_chunk.detach().numpy()[:16, :])
        # plt.show()


def create_chunks_parallel(
        rts: RealtimeSTFT,
        in_dir: str = RAW_AUDIO_DIR,
        out_dir: str = AUDIO_CHUNKS_PT_DIR,
        fx: Optional[Pedalboard] = None,
        fx_save_dir: Optional[str] = None,
        n_jobs: int = -1
) -> None:
    assert os.path.isdir(in_dir)
    if not os.path.isdir(out_dir):
        log.info(f'Making out_dir: {out_dir}')
        os.mkdir(out_dir)
    if fx_save_dir is not None:
        if not os.path.isdir(fx_save_dir):
            log.info(f'Making fx_save_dir: {fx_save_dir}')
            os.mkdir(fx_save_dir)

    audio_paths = []
    for root, dirs, files in os.walk(in_dir):
        for file_name in files:
            if file_name.endswith('.wav') and not file_name.startswith('.'):
                audio_paths.append(os.path.join(root, file_name))

    log.info(f'Found {len(audio_paths)} audio paths.')
    if n_jobs == 1:
        for path in tqdm(audio_paths):
            log.info(path)
            create_chunks(path, rts, out_dir, fx, fx_save_dir)
    else:
        joblib.Parallel(n_jobs=n_jobs, verbose=5)(
            joblib.delayed(create_chunks)(path, rts, out_dir, fx, fx_save_dir)
            for path in audio_paths
        )


if __name__ == '__main__':
    rts = RealtimeSTFT(
        n_fft=N_FFT,
        hop_len=HOP_LEN,
        model_io_n_frames=MODEL_IO_N_FRAMES,
        center=True
    )
    fx = None
    fx_save_dir = None
    # fx = Pedalboard([Distortion(drive_db=30.0)])
    # fx_save_dir = f'{AUDIO_CHUNKS_PT_DIR}__dist'

    create_chunks_parallel(
        rts,
        # in_dir=os.path.join(DATA_DIR, 'raw_eval'),
        in_dir=RAW_AUDIO_DIR,
        # out_dir=os.path.join(DATA_DIR, 'proc_eval'),
        # out_dir=AUDIO_CHUNKS_PT_DIR,
        out_dir=AUDIO_CHUNKS_PT_DIR,
        fx=fx,
        # fx_save_dir=os.path.join(DATA_DIR, 'fx_eval'),
        fx_save_dir=fx_save_dir,
        n_jobs=1,
    )
