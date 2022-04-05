import logging
import os
from typing import Union

import librosa as lr
import numpy as np
import soundfile as sf
import torch as tr
from numpy import ndarray
from torch import nn
from torch.jit import ScriptModule
from tqdm import tqdm

from config import SR, OUT_DIR, HOP_LEN, N_FFT, MODEL_IO_N_FRAMES
from modeling import SpecCNN2DSmall
from pl_wrapper import PLWrapper
from realtime_stft import RealtimeSTFT

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def process_file(path: str,
                 io_n_samples: int,
                 model: Union[nn.Module, ScriptModule],
                 save_suffix: str = '',
                 sr: int = SR,
                 mono: bool = False) -> (ndarray, ndarray):
    audio_in, _ = lr.load(path, sr=sr, mono=mono)
    if mono:
        audio_in = np.expand_dims(audio_in, axis=0)
    n_steps = audio_in.shape[-1] // io_n_samples
    audio_in = audio_in[:, :(n_steps * io_n_samples)]
    audio_pt = tr.tensor(audio_in)

    audio_out = []
    if hasattr(model, 'reset'):
        model.reset()
    for idx in tqdm(range(n_steps)):
        start_idx = idx * io_n_samples
        chunk_in = audio_pt[:, start_idx:start_idx + io_n_samples]
        chunk_out = model(chunk_in)
        audio_chunk = chunk_out.numpy()
        audio_out.append(audio_chunk)

    fade_samples = 0
    if hasattr(model, 'get_min_delay_samples'):
        fade_samples = model.get_min_delay_samples()
    elif hasattr(model, 'calc_min_delay_samples'):
        fade_samples = model.calc_min_delay_samples()

    if fade_samples > 0 and hasattr(model, 'flush'):
        audio_out.append(model.flush().numpy())

    audio_out = np.concatenate(audio_out, axis=-1)

    wav_name = os.path.basename(path)[:-4]
    in_save_name = f'{wav_name}__in__mono_{mono}__sr_{sr}.wav'
    sf.write(os.path.join(OUT_DIR, in_save_name),
             audio_in.T,
             samplerate=sr)
    out_save_name = f'{wav_name}{save_suffix}__out__mono_{mono}' \
                    f'__sr_{sr}__fade_{fade_samples}.wav'
    sf.write(os.path.join(OUT_DIR, out_save_name),
             audio_out.T,
             samplerate=sr)
    return audio_in, audio_out


if __name__ == '__main__':
    model = None
    model_path = None
    n_filters = None

    n_filters = 4
    # TODO(christhetree): check why center=False causes artifacts
    if SR == 44100:
        model_path = os.path.join(OUT_DIR, 'SpecCNN2DSmall__sr_44100__n_fft_2048__center_True__n_frames_16__pos_spec_False__n_filters_4__epoch=04__val_loss=0.298.ckpt')
    elif SR == 48000:
        model_path = os.path.join(OUT_DIR, 'SpecCNN2DSmall__sr_48000__n_fft_2048__center_True__n_frames_16__pos_spec_False__n_filters_4__epoch=04__val_loss=0.340.ckpt')
    model = SpecCNN2DSmall(n_filters=n_filters)

    if model_path:
        # This loads the weights into the model
        # We don't need the PLWrapper after this
        pl_wrapper = PLWrapper.load_from_checkpoint(
            model_path,
            model=model,
            rts=RealtimeSTFT(n_fft=N_FFT, hop_len=HOP_LEN),
            batch_size=1,
        )
        tr.save(model.state_dict(), f'{model_path[:-5]}.pt')

    # mono = True
    mono = False
    if mono:
        batch_size = 1
    else:
        batch_size = 2
    hop_len = HOP_LEN
    io_n_samples = 2048
    n_fft = N_FFT
    model_io_n_frames = MODEL_IO_N_FRAMES
    fade_n_samples = 32
    center = True
    # center = False
    spec_diff_mode = True
    # spec_diff_mode = False
    power = 1.0
    logarithmize = True
    # ensure_pos_spec = True
    ensure_pos_spec = False
    use_phase_info = True

    # Wrap the spectral model with an RTS
    rts = RealtimeSTFT(
        model=model,
        batch_size=batch_size,
        io_n_samples=io_n_samples,
        n_fft=n_fft,
        hop_len=hop_len,
        model_io_n_frames=model_io_n_frames,
        center=center,
        spec_diff_mode=spec_diff_mode,
        power=power,
        logarithmize=logarithmize,
        ensure_pos_spec=ensure_pos_spec,
        use_phase_info=use_phase_info,
        fade_n_samples=fade_n_samples,
    )

    audio_dir = '/Users/puntland/local_christhetree/qosmo/residual_audio/data/raw_eval'
    audio_paths = [os.path.join(audio_dir, _) for _ in os.listdir(audio_dir)
                   if _.endswith('.wav')]
    # for path in audio_paths:
    #     audio_in, audio_out = process_file(
    #         path,
    #         rts,
    #         save_suffix=f'__{model.__class__.__name__}__python'
    #     )
    #     if fade_n_samples == 0:
    #         assert len(audio_in) == len(audio_out)
    #         diff = np.abs(audio_in - audio_out)
    #         log.info(f'diff mean = {np.mean(diff)}')
    #         log.info(f'diff max = {np.max(diff)}')
    #     exit()

    tmp_script = tr.jit.script(rts.eval())
    tr.jit.save(tmp_script, os.path.join(OUT_DIR, 'tmp.pt'))
    script = tr.jit.load(os.path.join(OUT_DIR, 'tmp.pt'))
    script = tr.jit.freeze(
        script,
        preserved_attrs=['calc_min_delay_samples', 'reset', 'flush']
    )
    # script = tr.jit.optimize_for_inference(script)
    # exit()
    # Neutone test
    # script = tr.jit.load(os.path.join(OUT_DIR, 'model.pt'))

    for idx, path in enumerate(audio_paths):
        # io_n_samples = (idx + 1) * 512
        # log.info(f'io_n_samples = {io_n_samples}')
        # script.set_buffer_size(io_n_samples)
        process_file(path,
                     io_n_samples,
                     script,
                     save_suffix=f'__{model.__class__.__name__}'
                                 f'__n_filters_{n_filters}'
                                 f'__sdm_{spec_diff_mode}',
                     mono=mono)
        # exit()
