import logging
import os

import numpy as np
from torch import Tensor as T
import torch as tr
from torch import nn
from torchaudio.transforms import Spectrogram
import matplotlib.pyplot as plt

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class RealtimeSTFT(nn.Module):
    def __init__(self,
                 batch_size: int,
                 io_n_samples: int = 512,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 model_io_n_frames: int = 16,
                 crossfade_n_frames: int = 1,
                 eps: float = 1e-7) -> None:
        super().__init__()
        assert io_n_samples >= hop_length
        assert io_n_samples % hop_length == 0
        assert n_fft % 2 == 0
        assert n_fft % hop_length == 0
        self.batch_size = batch_size
        self.io_n_samples = io_n_samples
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.model_io_n_frames = model_io_n_frames
        self.crossfade_n_frames = crossfade_n_frames
        self.eps = eps

        self._io_n_frames = io_n_samples // hop_length
        self._overlap = self.n_fft // 2
        self._overlap_n_frames = self._overlap // self.hop_length

        self.stft = Spectrogram(n_fft=self.n_fft,
                                hop_length=self.hop_length,
                                pad=0,
                                center=True,
                                normalized=False,
                                power=2.0,
                                return_complex=True)
        self._in_buf = None
        self._tmp_in_buf = None
        self._frames_buf = None
        self._tmp_frames_buf = None
        self.reset()

    def reset(self) -> None:
        self._in_buf = tr.full(
            (self.batch_size, self.n_fft + self.io_n_samples),
            self.eps,
        )
        self._tmp_in_buf = tr.clone(self._in_buf)
        # self._frames_buf = self.stft(tr.full(
        #     (self.batch_size, (self.model_io_n_frames * self.hop_length) - 1),
        #     self.eps)
        # )
        self._frames_buf = tr.zeros((self.batch_size, 5, self.model_io_n_frames))
        self._tmp_frames_buf = tr.clone(self._frames_buf)

    def audio_to_spectrogram(self, audio: T) -> T:
        assert audio.shape == (self.batch_size, self.io_n_samples)
        self._tmp_in_buf[:, :-self.io_n_samples] = self._in_buf[:, self.io_n_samples:]
        self._in_buf[:, :-self.io_n_samples] = self._tmp_in_buf[:, :-self.io_n_samples]
        self._in_buf[:, -self.io_n_samples:] = audio
        print(f'in_buf = {self._in_buf}')
        frames = self.stft(self._in_buf)
        frames = frames[:, :, self._overlap_n_frames:]
        fixed_prev_frames = frames[:, :, :-self._io_n_frames]
        n_fixed_prev_frames = fixed_prev_frames.shape[2]
        new_frames = frames[:, :, -self._io_n_frames:]
        self._frames_buf[:, :, -n_fixed_prev_frames:] = fixed_prev_frames
        self._tmp_frames_buf[:, :, :-self._io_n_frames] = self._frames_buf[:, :, self._io_n_frames:]
        self._frames_buf[:, :, :-self._io_n_frames] = self._tmp_frames_buf[:, :, :-self._io_n_frames]
        self._frames_buf[:, :, -self._io_n_frames:] = new_frames
        print(f'frames_buf = {self._frames_buf[:, -1, :]}')
        return self._frames_buf

    def forward(self, audio: T) -> T:
        return self.audio_to_spectrogram(audio)


if __name__ == '__main__':
    batch_size = 1
    io_n_samples = 2
    n_fft = 8
    hop_length = 2
    model_io_n_frames = 19
    audio_n_frames = 23

    audio = tr.rand((batch_size, hop_length * audio_n_frames))
    rts = RealtimeSTFT(
        batch_size, io_n_samples, n_fft, hop_length, model_io_n_frames)

    all_at_once = rts.stft(audio)
    chunked_out = None

    n_steps = (hop_length * audio_n_frames) // io_n_samples
    for idx in range(n_steps):
        start_idx = idx * io_n_samples
        chunk = audio[:, start_idx:start_idx + io_n_samples]
        chunked_out = rts(chunk)

    all_np = all_at_once.numpy()[0]
    chunked_np = chunked_out.numpy()[0]
    all_np = all_np[:, -model_io_n_frames:]
    # exit()
    print(np.allclose(all_np, chunked_np))
    print(np.allclose(all_np[:, 1:], chunked_np[:, 1:]))

    plt.imshow(all_np)
    plt.title('all_np')
    plt.show()
    plt.imshow(chunked_np)
    plt.title('chunked_np')
    plt.show()
    derp = 1
