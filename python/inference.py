import logging
import os

import librosa as lr
import numpy as np
import soundfile as sf
import torch as tr
from tqdm import tqdm

from config import SR, OUT_DIR
from modeling import SpecCNN2D
from pl_wrapper import PLWrapper
from realtime_stft import RealtimeSTFT

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def process_file(path: str,
                 rts: RealtimeSTFT,
                 save_suffix: str = '',
                 sr: int = SR) -> None:
    audio_in, _ = lr.load(path, sr=sr, mono=True)
    n_steps = len(audio_in) // rts.io_n_samples
    audio_in = audio_in[:n_steps * rts.io_n_samples]
    audio_pt = tr.tensor(audio_in).unsqueeze(dim=0)

    audio_out = []
    rts.reset()
    for idx in tqdm(range(n_steps)):
        start_idx = idx * rts.io_n_samples
        chunk_in = audio_pt[:, start_idx:start_idx + io_n_samples]
        chunk_out = rts(chunk_in)
        audio_chunk = chunk_out[0].numpy()
        audio_out.append(audio_chunk)

    audio_out.append(rts.flush()[0].numpy())
    audio_out = np.concatenate(audio_out)

    wav_name = os.path.basename(path)[:-4]
    in_save_name = f'{wav_name}__in.wav'
    sf.write(os.path.join(OUT_DIR, in_save_name),
             audio_in,
             samplerate=sr)
    out_save_name = f'{wav_name}{save_suffix}__out__fade_{rts.fade_n_samples}.wav'
    sf.write(os.path.join(OUT_DIR, out_save_name),
             audio_out,
             samplerate=sr)


if __name__ == '__main__':
    # model = None
    model = SpecCNN2D()

    model_path = os.path.join(OUT_DIR, 'testing__epoch=05__val_loss=0.289.ckpt')
    pl_wrapper = PLWrapper.load_from_checkpoint(
        model_path,
        model=model,
        rts=RealtimeSTFT(),
        batch_size=1,
    )

    batch_size = 1
    hop_length = 512
    io_n_samples = 512
    n_fft = 2048
    model_io_n_frames = 16
    fade_n_samples = 32
    power = 1.0
    logarithmize = True
    use_phase_info = True

    rts = RealtimeSTFT(
        model,
        batch_size,
        io_n_samples,
        n_fft,
        hop_length,
        model_io_n_frames,
        power,
        logarithmize,
        use_phase_info,
        fade_n_samples,
    )
    scripted = tr.jit.script(rts)
    tr.jit.save(scripted, os.path.join(OUT_DIR, 'testing.pt'))
    scripted_2 = tr.jit.load(os.path.join(OUT_DIR, 'testing.pt'))
    # frozen = tr.jit.freeze(scripted_2.eval(), preserved_attrs=['io_n_samples', 'reset', 'flush', 'fade_n_samples'])
    # frozen = tr.jit.optimize_for_inference(frozen)
    # exit()

    audio_dir = '/Users/puntland/local_christhetree/qosmo/residual_audio/data/raw_eval'
    audio_paths = [os.path.join(audio_dir, _) for _ in os.listdir(audio_dir)
                   if _.endswith('.wav')]
    for path in audio_paths:
        process_file(path, rts, save_suffix='__rts')
    for path in audio_paths:
        process_file(path, scripted_2, save_suffix='__script')
