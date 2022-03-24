import logging
import os

import librosa as lr
import numpy as np
import soundfile as sf
import torch as tr
from tqdm import tqdm

from config import SR, OUT_DIR, HOP_LEN, N_FFT
from modeling import SpecCNN2DSmall, SpecCNN1D, SpecCNN2DLarge, SpecCNN2DMinimal
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
        wet_ratio = 1.0  # TODO(christhetree): make dynamic
        # wet_ratio = idx / int(0.5 * n_steps)
        # wet_ratio = min(1.0, 0.0 + wet_ratio)
        start_idx = idx * rts.io_n_samples
        chunk_in = audio_pt[:, start_idx:start_idx + io_n_samples]
        chunk_out = rts(chunk_in, wet_ratio)
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
    model = None
    model_path = None
    n_filters = None

    n_filters = 4
    model_path = os.path.join(OUT_DIR, 'SpecCNN2D__testing__epoch=05__val_loss=0.296.ckpt')
    model = SpecCNN2DSmall(n_filters=n_filters)

    # n_filters = 16
    # model_path = os.path.join(OUT_DIR, 'SpecCNN2D__n_fft_1024__n_filters_16__epoch=04__val_loss=0.262.ckpt')
    # model = SpecCNN2DLarge(n_filters=n_filters)

    # n_filters = 64
    # model_path = os.path.join(OUT_DIR, 'SpecCNN1D__n_fft_2048__n_filters_64__epoch=04__val_loss=0.300.ckpt')
    # n_filters = 4
    # model_path = os.path.join(OUT_DIR, 'SpecCNN1D__n_fft_2048__n_filters_4__epoch=04__val_loss=0.498.ckpt')
    # model = SpecCNN1D(n_filters=n_filters)

    if model_path:
        # This loads the weights into the model
        # We don't need the PLWrapper after this
        pl_wrapper = PLWrapper.load_from_checkpoint(
            model_path,
            model=model,
            rts=RealtimeSTFT(n_fft=N_FFT, hop_len=HOP_LEN),
            batch_size=1,
        )

    batch_size = 1
    hop_length = HOP_LEN
    io_n_samples = 1024
    n_fft = N_FFT
    model_io_n_frames = 16
    fade_n_samples = 32
    spec_diff_mode = True
    # spec_diff_mode = False
    power = 1.0
    logarithmize = True
    use_phase_info = True

    # Wrap the spectral model with an RTS
    rts = RealtimeSTFT(
        model,
        batch_size,
        io_n_samples,
        n_fft,
        hop_length,
        model_io_n_frames,
        spec_diff_mode,
        power,
        logarithmize,
        use_phase_info,
        fade_n_samples,
    )

    audio_dir = '/Users/puntland/local_christhetree/qosmo/residual_audio/data/raw_eval'
    audio_paths = [os.path.join(audio_dir, _) for _ in os.listdir(audio_dir)
                   if _.endswith('.wav')]
    # for path in audio_paths:
    #     process_file(path, rts, save_suffix=f'__{model.__class__.__name__}__python')
    #     exit()

    scripted = tr.jit.script(rts.eval())
    tr.jit.save(scripted, os.path.join(OUT_DIR, 'tmp.pt'))
    scripted_2 = tr.jit.load(os.path.join(OUT_DIR, 'tmp.pt'))
    frozen = tr.jit.freeze(
        scripted_2,
        preserved_attrs=['io_n_samples', 'reset', 'flush', 'fade_n_samples']
    )
    frozen = tr.jit.optimize_for_inference(frozen)
    # exit()

    for path in audio_paths:
        process_file(path, frozen, save_suffix=f'__{model.__class__.__name__}'
                                               f'__n_filters_{n_filters}'
                                               f'__sdm_{spec_diff_mode}')
        # exit()
