import logging
import os

import joblib
import librosa as lr
import torch as tr
from tqdm import tqdm

from config import N_FFT, HOP_LEN, AUDIO_CHUNKS_PT_DIR, SR, RAW_AUDIO_DIR
from realtime_stft import RealtimeSTFT

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def create_chunks(path: str,
                  rts: RealtimeSTFT,
                  save_dir: str,
                  overlap: float = 0.5,
                  sr: int = SR) -> None:
    try:
        audio, _ = lr.load(path, sr=sr, mono=True)
    except:
        log.warning(f'Failed to load: {path}')
        return

    audio_len = len(audio)
    proc_window_len = (rts.model_io_n_frames - 1) * rts.hop_len
    proc_hop_len = int(proc_window_len * overlap)
    n_steps = ((audio_len - proc_window_len) // proc_hop_len) + 1
    file_name = os.path.basename(path)

    for idx in range(n_steps):
        start_idx = idx * proc_hop_len
        end_idx = start_idx + proc_window_len
        audio_chunk = audio[start_idx:end_idx]
        assert len(audio_chunk) == proc_window_len
        audio_pt = tr.tensor(audio_chunk)
        save_path = os.path.join(save_dir, f'{file_name}__{idx:04}.pt')
        tr.save(audio_pt, save_path)

        # spec_chunk = rts.audio_to_spec_offline(audio_pt.unsqueeze(0))
        # import matplotlib.pyplot as plt
        # plt.imshow(spec_chunk.detach().numpy()[:16, :])
        # plt.show()


def create_chunks_parallel(
        rts: RealtimeSTFT,
        in_dir: str = RAW_AUDIO_DIR,
        out_dir: str = AUDIO_CHUNKS_PT_DIR,
        n_jobs: int = -1
) -> None:
    audio_paths = []
    for root, dirs, files in os.walk(in_dir):
        for file_name in files:
            if file_name.endswith('.wav') and not file_name.startswith('.'):
                audio_paths.append(os.path.join(root, file_name))

    log.info(f'Found {len(audio_paths)} audio paths.')
    if n_jobs == 1:
        for path in tqdm(audio_paths):
            log.info(path)
            create_chunks(path, rts, out_dir)
    else:
        joblib.Parallel(n_jobs=n_jobs, verbose=5)(
            joblib.delayed(create_chunks)(path, rts, out_dir)
            for path in audio_paths
        )


if __name__ == '__main__':
    rts = RealtimeSTFT(n_fft=N_FFT, hop_len=HOP_LEN)
    create_chunks_parallel(
        rts,
        n_jobs=1,
    )
