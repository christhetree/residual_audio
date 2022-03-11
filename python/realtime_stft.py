import logging
import os
from typing import Optional

import librosa as lr
import numpy as np
import soundfile as sf
import torch as tr
from torch import Tensor as T
from torch import nn
from torchaudio.transforms import Spectrogram, InverseSpectrogram, GriffinLim
from tqdm import tqdm

from python.config import OUT_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

SR = 44100


class RealtimeSTFT(nn.Module):
    def __init__(self,
                 batch_size: int = 1,
                 io_n_samples: int = 512,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 model_io_n_frames: int = 16,
                 power: Optional[float] = None,
                 use_phase_info: bool = True,
                 fade_n_samples: int = 0,
                 eps: float = 1e-7) -> None:
        super().__init__()
        assert io_n_samples >= hop_length
        assert io_n_samples % hop_length == 0
        assert n_fft % 2 == 0
        assert (n_fft // 2) % hop_length == 0
        assert power is None or power >= 1.0
        assert fade_n_samples <= io_n_samples
        self.batch_size = batch_size
        self.io_n_samples = io_n_samples
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.model_io_n_frames = model_io_n_frames
        self.power = power
        self.use_phase_info = use_phase_info
        self.fade_n_samples = fade_n_samples
        self.eps = eps

        self.io_n_frames = self.io_n_samples // self.hop_length
        self.overlap_n_frames = self.n_fft // 2 // self.hop_length
        self.in_buf_n_frames = (self.overlap_n_frames * 2) - 1 + self.io_n_frames
        self.n_bins = (self.n_fft // 2) + 1
        self.stft_out_shape = (self.batch_size, self.n_bins, self.in_buf_n_frames + 1)
        self.model_io_shape = (self.batch_size, self.n_bins, self.model_io_n_frames)
        self.out_buf_n_samples = self.io_n_samples + self.fade_n_samples
        assert self.out_buf_n_samples <= (self.in_buf_n_frames - 1) * self.hop_length

        self.stft = Spectrogram(n_fft=self.n_fft,
                                hop_length=self.hop_length,
                                pad=0,
                                center=True,
                                normalized=False,
                                power=None,
                                return_complex=True)
        self.istft = InverseSpectrogram(n_fft=self.n_fft,
                                        hop_length=self.hop_length,
                                        pad=0,
                                        center=True,
                                        normalized=False)

        self.in_buf = tr.full(
            (self.batch_size, self.in_buf_n_frames * self.hop_length),
            self.eps,
        )
        self.tmp_in_buf = tr.clone(self.in_buf)

        self.stft_mag_buf = tr.full(self.stft_out_shape, self.eps)
        self.mag_buf = tr.full(self.model_io_shape, self.eps)
        self.tmp_mag_buf = tr.clone(self.mag_buf)

        self.stft_phase_buf = tr.zeros(self.stft_out_shape)
        self.phase_buf = tr.zeros(self.model_io_shape)
        self.tmp_phase_buf = tr.clone(self.phase_buf)

        self.out_frames_buf = tr.full(
            (self.batch_size, self.n_bins, self.in_buf_n_frames),
            self.eps,
            dtype=tr.complex64,
        )
        self.out_buf = tr.full(
            (self.batch_size, self.out_buf_n_samples),
            self.eps,
        )
        self.reset()

        self.fade_up = None
        self.fade_down = None
        if self.fade_n_samples > 0:
            self.fade_up = tr.linspace(0, 1, self.fade_n_samples)
            self.fade_down = tr.linspace(1, 0, self.fade_n_samples)
        self.zero_phase = tr.zeros(self.model_io_shape)

    @property
    def min_delay_samples(self) -> int:
        return self.fade_n_samples

    def reset(self) -> None:
        self.in_buf.fill_(self.eps)
        self.tmp_in_buf.fill_(self.eps)
        self.stft_mag_buf.fill_(self.eps)
        self.mag_buf.fill_(self.eps)
        self.tmp_mag_buf.fill_(self.eps)
        self.stft_phase_buf.fill_(0)
        self.phase_buf.fill_(0)
        self.tmp_phase_buf.fill_(0)
        self.out_frames_buf.fill_(self.eps)
        self.out_buf.fill_(self.eps)

    def _update_mag_or_phase_buffers(self,
                                     stft_out_buf: T,
                                     frames_buf: T,
                                     tmp_frames_buf: T) -> None:
        # Remove overlap frames we have computed before
        frames = stft_out_buf[:, :, self.overlap_n_frames:]
        # Identify frames that are more correct due to missing prev audio info
        fixed_prev_frames = frames[:, :, :-self.io_n_frames]
        assert fixed_prev_frames.shape[2] == self.overlap_n_frames
        # Identify the new frames for the input audio chunk
        new_frames = frames[:, :, -self.io_n_frames:]
        # Overwrite previous frames with more correct frames
        frames_buf[:, :, -self.overlap_n_frames:] = fixed_prev_frames
        # Shift buffer left and insert new frames without allocating memory
        tmp_frames_buf[:, :, :-self.io_n_frames] = frames_buf[:, :, self.io_n_frames:]
        frames_buf[:, :, :-self.io_n_frames] = tmp_frames_buf[:, :, :-self.io_n_frames]
        frames_buf[:, :, -self.io_n_frames:] = new_frames

    def audio_to_spec(self, audio: T) -> T:
        assert audio.shape == (self.batch_size, self.io_n_samples)
        # Shift buffer left and insert audio chunk without allocating memory
        self.tmp_in_buf[:, :-self.io_n_samples] = self.in_buf[:, self.io_n_samples:]
        self.in_buf[:, :-self.io_n_samples] = self.tmp_in_buf[:, :-self.io_n_samples]
        self.in_buf[:, -self.io_n_samples:] = audio

        complex_frames = self.stft(self.in_buf)
        if self.power is None:
            self.stft_mag_buf = complex_frames.real
        else:
            tr.abs(complex_frames, out=self.stft_mag_buf)
            if self.power != 1.0:
                tr.pow(self.stft_mag_buf, self.power, out=self.stft_mag_buf)
        self._update_mag_or_phase_buffers(
            self.stft_mag_buf, self.mag_buf, self.tmp_mag_buf)

        if self.use_phase_info:
            if self.power is None:
                self.stft_phase_buf = complex_frames.imag
            else:
                tr.angle(complex_frames, out=self.stft_phase_buf)
            self._update_mag_or_phase_buffers(
                self.stft_phase_buf, self.phase_buf, self.tmp_phase_buf)

        return self.mag_buf

    def spec_to_audio(self, spec: T) -> T:
        assert spec.shape == self.model_io_shape
        spec = spec[:, :, -self.in_buf_n_frames:]
        if self.use_phase_info:
            phase = self.phase_buf[:, :, -self.in_buf_n_frames:]
        else:
            phase = self.zero_phase[:, :, -self.in_buf_n_frames:]

        if self.power is None:
            self.out_frames_buf.real = spec
            self.out_frames_buf.imag = phase
        else:
            if self.power != 1.0:
                tr.pow(spec, 1 / self.power, out=spec)
            tr.polar(spec, phase, out=self.out_frames_buf)

        rec_audio = self.istft(self.out_frames_buf)
        rec_audio = rec_audio[:, -self.out_buf_n_samples:]
        if self.fade_n_samples == 0:
            return rec_audio

        self.out_buf[:, -self.fade_n_samples:] *= self.fade_down
        rec_audio[:, :self.fade_n_samples] *= self.fade_up
        rec_audio[:, :self.fade_n_samples] += self.out_buf[:, -self.fade_n_samples:]
        audio_out = rec_audio[:, :self.io_n_samples]
        self.out_buf = rec_audio
        return audio_out

    def flush(self) -> T:
        # TODO(christhetree): prevent this memory allocation
        audio_out = tr.full((self.batch_size, self.io_n_samples), self.eps)
        if self.fade_n_samples > 0:
            audio_out[:, :self.fade_n_samples] = self.out_buf[:, -self.fade_n_samples:]
        else:
            log.warning('Flushing is not necessary when fade_n_samples == 0')
        return audio_out

    def forward(self, audio: T) -> (T, T):
        spec = self.audio_to_spec(audio)
        rec_audio = self.spec_to_audio(spec)
        return spec, rec_audio


def process_file(path: str, rts: RealtimeSTFT, sr: int = SR) -> None:
    audio_in, _ = lr.load(path, sr=sr, mono=True)
    n_steps = len(audio_in) // rts.io_n_samples
    audio_in = audio_in[:n_steps * rts.io_n_samples]
    audio_pt = tr.tensor(audio_in).unsqueeze(dim=0)

    audio_out = []
    rts.reset()
    for idx in tqdm(range(n_steps)):
        start_idx = idx * rts.io_n_samples
        chunk_in = audio_pt[:, start_idx:start_idx + io_n_samples]
        _, chunk_out = rts(chunk_in)
        audio_chunk = chunk_out[0].numpy()
        audio_out.append(audio_chunk)

    audio_out.append(rts.flush()[0].numpy())
    audio_out = np.concatenate(audio_out)

    wav_name = os.path.basename(path)[:-4]
    in_save_name = f'{wav_name}__in.wav'
    sf.write(os.path.join(OUT_DIR, in_save_name),
             audio_in,
             samplerate=sr)
    out_save_name = f'{wav_name}__out.wav'
    sf.write(os.path.join(OUT_DIR, out_save_name),
             audio_out,
             samplerate=sr)


if __name__ == '__main__':
    batch_size = 1
    hop_length = 512
    io_n_samples = 512
    n_fft = 2048
    model_io_n_frames = 16
    fade_n_samples = 0
    power = 2.0
    use_phase_info = True

    rts = RealtimeSTFT(
        batch_size,
        io_n_samples,
        n_fft,
        hop_length,
        model_io_n_frames,
        power,
        use_phase_info,
        fade_n_samples
    )

    audio_dir = '/Users/puntland/local_christhetree/qosmo/residual_audio/data/raw_eval'
    audio_paths = [os.path.join(audio_dir, _) for _ in os.listdir(audio_dir) 
                   if _.endswith('.wav')]
    for path in audio_paths:
        process_file(path, rts)

    # audio_n_frames = 16
    # audio = tr.rand((batch_size, hop_length * audio_n_frames))
    # assert audio_n_frames % rts.io_n_frames == 0
    #
    # all_spec = rts.stft(audio)
    #
    # chunked_spec = None
    # audio_out = []
    # n_steps = (hop_length * audio_n_frames) // io_n_samples
    # for idx in range(n_steps):
    #     start_idx = idx * io_n_samples
    #     chunk_in = audio[:, start_idx:start_idx + io_n_samples]
    #     chunked_spec, chunk_out = rts(chunk_in)
    #     # print(np.allclose(chunk_in.numpy(), chunk_out.numpy()))
    #     audio_out.append(chunk_out.numpy())
    #
    # audio_out = np.concatenate(audio_out, axis=1)
    #
    # all_np = all_spec.numpy()[0]
    # chunked_np = chunked_spec.numpy()[0]
    #
    # cut_idx = rts.overlap_n_frames - 1
    # cut_chunked_np = chunked_np[:, cut_idx:]
    # all_np = all_np[:, -model_io_n_frames:]
    # cut_all_np = all_np[:, cut_idx:]
    # print(np.allclose(all_np, chunked_np))
    # print(np.allclose(cut_all_np, cut_chunked_np))
    #
    # import matplotlib.pyplot as plt
    # # plt.imshow(all_np)
    # plt.imshow(cut_all_np)
    # plt.title('all_np')
    # plt.show()
    # # plt.imshow(chunked_np)
    # plt.imshow(cut_chunked_np)
    # plt.title('chunked_np')
    # plt.show()
