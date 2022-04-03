import logging
import os
from typing import Optional, List

import torch as tr
from torch import Tensor
from torch import nn

from config import EPS, MODEL_IO_N_FRAMES

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class RealtimeSTFT(nn.Module):
    def __init__(self,
                 model: Optional[nn.Module] = None,
                 batch_size: int = 1,
                 io_n_samples: int = 512,
                 n_fft: int = 2048,
                 hop_len: int = 512,
                 window: Optional[Tensor] = None,
                 model_io_n_frames: int = MODEL_IO_N_FRAMES,
                 spec_diff_mode: bool = False,
                 center: bool = False,
                 power: Optional[float] = 1.0,
                 logarithmize: bool = True,
                 ensure_pos_spec: bool = True,
                 use_phase_info: bool = True,
                 fade_n_samples: int = 0,
                 eps: float = EPS) -> None:
        super().__init__()
        assert n_fft % 2 == 0
        assert (n_fft // 2) % hop_len == 0
        if window is not None:
            assert window.shape == (n_fft,)
        if center:
            log.warning('STFT is not causal when center=True')
        assert power is None or power >= 1.0
        if power is not None and power > 1.0:
            log.warning('A power greater than 1.0 probably adds unnecessary '
                        'computational complexity')
        assert eps < 1.0
        self.model = model
        self.batch_size = batch_size
        self.io_n_samples = io_n_samples
        self.n_fft = n_fft
        self.hop_len = hop_len
        if window is None:
            window = tr.hann_window(self.n_fft)
            if not center:
                # Ensures the NOLA constraint is met for the hann_window
                # See https://github.com/pytorch/pytorch/issues/62323
                # 1e-5 is chosen based on the torchaudio implementation
                window = tr.clamp(window, min=1e-5)
        self.window = window
        self.model_io_n_frames = model_io_n_frames
        self.spec_diff_mode = spec_diff_mode
        self.center = center
        self.power = power
        self.logarithmize = logarithmize
        self.ensure_pos_spec = ensure_pos_spec
        self.use_phase_info = use_phase_info
        self.fade_n_samples = fade_n_samples
        self.eps = eps

        # Internal parameters
        self.io_n_frames = None
        self.overlap_n_frames = None
        self.in_buf_n_frames = None
        self.n_bins = None
        self.stft_out_shape = None
        self.istft_in_n_frames = None
        self.istft_length = None
        self.model_io_shape = None
        self.out_buf_n_samples = None
        self.log10_eps = None

        # Internal buffers
        self.in_buf = None
        self.stft_mag_buf = None
        self.mag_buf = None
        self.spec_out_buf = None
        self.stft_phase_buf = None
        self.phase_buf = None
        self.out_frames_buf = None
        self.out_buf = None

        # Sets internal parameters and allocates buffers
        self.set_buffer_size(io_n_samples)

        # These must be instantiated for TorchScript
        self.fade_up = tr.linspace(0, 1, max(self.fade_n_samples, 1))
        self.fade_down = tr.linspace(1, 0, max(self.fade_n_samples, 1))
        self.zero_phase = tr.zeros(self.model_io_shape)

    def _set_internal_params(self) -> None:
        self.io_n_frames = self.io_n_samples // self.hop_len
        assert self.io_n_frames <= self.model_io_n_frames
        self.overlap_n_frames = self.n_fft // 2 // self.hop_len
        self.in_buf_n_frames = (2 * self.overlap_n_frames) + self.io_n_frames - 1
        self.n_bins = (self.n_fft // 2) + 1
        if self.center:
            self.stft_out_shape = (
                self.batch_size,
                self.n_bins,
                (2 * self.overlap_n_frames) + self.io_n_frames,
            )
            self.istft_in_n_frames = self.overlap_n_frames + self.io_n_frames
            self.istft_length = (self.istft_in_n_frames - 1) * self.hop_len
        else:
            self.stft_out_shape = (self.batch_size,
                                   self.n_bins,
                                   self.io_n_frames)
            self.istft_in_n_frames = self.io_n_frames
            self.istft_length = self.in_buf_n_frames * self.hop_len
        assert self.istft_in_n_frames <= self.model_io_n_frames

        self.model_io_shape = (self.batch_size,
                               self.n_bins,
                               self.model_io_n_frames)
        self.out_buf_n_samples = self.io_n_samples + self.fade_n_samples
        assert self.out_buf_n_samples <= self.istft_length
        self.log10_eps = tr.log10(tr.tensor([self.eps]))

    def _allocate_buffers(self) -> None:
        self.in_buf = tr.full(
            (self.batch_size, self.in_buf_n_frames * self.hop_len),
            self.eps,
        )

        self.stft_mag_buf = tr.full(self.stft_out_shape, self.eps)
        self.mag_buf = tr.full(self.model_io_shape, self.eps)
        # Required to allow inplace operations after the encoder
        self.spec_out_buf = tr.clone(self.mag_buf)

        self.stft_phase_buf = tr.zeros(self.stft_out_shape)
        self.phase_buf = tr.zeros(self.model_io_shape)

        self.out_frames_buf = tr.full(
            (self.batch_size, self.n_bins, self.istft_in_n_frames),
            self.eps,
            dtype=tr.complex64,
        )
        self.out_buf = tr.full(
            (self.batch_size, self.out_buf_n_samples),
            self.eps,
        )

    @tr.jit.export
    def set_buffer_size(self, io_n_samples: int) -> None:
        assert io_n_samples >= self.hop_len
        assert io_n_samples % self.hop_len == 0
        assert self.fade_n_samples <= io_n_samples
        self.io_n_samples = io_n_samples
        self._set_internal_params()
        self._allocate_buffers()
        self.reset()

    @tr.jit.export
    def calc_min_delay_samples(self) -> int:
        return self.fade_n_samples

    @tr.jit.export
    def calc_min_buffer_size(self) -> int:
        return self.hop_len

    @tr.jit.export
    def calc_max_buffer_size(self) -> int:
        return self.model_io_n_frames * self.hop_len

    @tr.jit.export
    def calc_supported_buffer_sizes(self) -> List[int]:
        min_buffer_size = self.calc_min_buffer_size()
        max_buffer_size = self.calc_max_buffer_size()
        buffer_sizes = [bs for bs in range(min_buffer_size,
                                           max_buffer_size + 1,
                                           self.hop_len)]
        return buffer_sizes

    @tr.jit.export
    def reset(self) -> None:
        self.in_buf.fill_(self.eps)
        self.stft_mag_buf.fill_(self.eps)
        self.mag_buf.fill_(self.eps)
        self.spec_out_buf.fill_(self.eps)
        self.stft_phase_buf.fill_(0)
        self.phase_buf.fill_(0)
        self.out_frames_buf.fill_(self.eps)
        self.out_buf.fill_(self.eps)

    def _update_mag_or_phase_buffers(self,
                                     stft_out_buf: Tensor,
                                     frames_buf: Tensor) -> Tensor:
        if self.center:
            # Remove overlap frames we have computed before
            frames = stft_out_buf[:, :, self.overlap_n_frames:]
            # Identify frames that are more correct due to missing prev audio
            fixed_prev_frames = frames[:, :, :-self.io_n_frames]
            assert fixed_prev_frames.shape[2] == self.overlap_n_frames
            # Identify the new frames for the input audio chunk
            new_frames = frames[:, :, -self.io_n_frames:]
            # Overwrite previous frames with more correct frames
            n_fixed_frames = min(self.model_io_n_frames, self.overlap_n_frames)
            frames_buf[:, :, -n_fixed_frames:] = fixed_prev_frames[:, :, -n_fixed_frames:]
        else:
            new_frames = stft_out_buf[:, :, -self.io_n_frames:]

        # Shift buffer left and insert new frames
        frames_buf = tr.roll(frames_buf, -self.io_n_frames, dims=2)
        frames_buf[:, :, -self.io_n_frames:] = new_frames
        return frames_buf

    @tr.jit.ignore
    def audio_to_spec_offline(self, audio: Tensor) -> Tensor:
        assert audio.shape[0] == self.batch_size
        assert audio.shape[1] >= self.n_fft
        assert audio.shape[1] % self.hop_len == 0
        spec = tr.stft(audio,
                       n_fft=self.n_fft,
                       hop_length=self.hop_len,
                       window=self.window,
                       center=self.center,
                       return_complex=True)
        if self.power is None:
            spec = spec.real
        else:
            spec = spec.abs()
            if self.power != 1.0:
                spec = spec.pow(self.power)

        if self.logarithmize:
            spec = tr.clamp(spec, min=self.eps)
            spec = tr.log10(spec)
            if self.ensure_pos_spec:
                spec -= self.log10_eps

        return spec

    @tr.jit.export
    def audio_to_spec(self, audio: Tensor) -> Tensor:
        assert audio.shape == (self.batch_size, self.io_n_samples)
        # Shift buffer left and insert audio chunk
        self.in_buf = tr.roll(self.in_buf, -self.io_n_samples, dims=1)
        self.in_buf[:, -self.io_n_samples:] = audio

        complex_frames = tr.stft(self.in_buf,
                                 n_fft=self.n_fft,
                                 hop_length=self.hop_len,
                                 window=self.window,
                                 center=self.center,
                                 return_complex=True)
        if self.power is None:
            self.stft_mag_buf = complex_frames.real
        else:
            tr.abs(complex_frames, out=self.stft_mag_buf)
            if self.power != 1.0:
                tr.pow(self.stft_mag_buf, self.power, out=self.stft_mag_buf)
        if self.logarithmize:
            self._logarithmize_spec(self.stft_mag_buf)
            if self.ensure_pos_spec:
                self.stft_mag_buf -= self.log10_eps

        self.mag_buf = self._update_mag_or_phase_buffers(self.stft_mag_buf,
                                                         self.mag_buf)

        if self.use_phase_info:
            if self.power is None:
                self.stft_phase_buf = complex_frames.imag
            else:
                tr.angle(complex_frames, out=self.stft_phase_buf)
            self.phase_buf = self._update_mag_or_phase_buffers(
                self.stft_phase_buf, self.phase_buf)

        # Prevent future inplace operations from mutating self.mag_buf
        self.spec_out_buf[:, :] = self.mag_buf
        return self.spec_out_buf

    @tr.jit.export
    def spec_to_audio(self, spec: Tensor) -> Tensor:
        assert spec.shape == self.model_io_shape
        spec = spec[:, :, -self.istft_in_n_frames:]
        if self.use_phase_info:
            phase = self.phase_buf[:, :, -self.istft_in_n_frames:]
        else:
            phase = self.zero_phase[:, :, -self.istft_in_n_frames:]

        if self.logarithmize:
            if self.ensure_pos_spec:
                spec += self.log10_eps
            self._unlogarithmize_spec(spec)

        if self.power is None:
            self.out_frames_buf.real = spec
            self.out_frames_buf.imag = phase
        else:
            if self.power != 1.0:
                tr.pow(spec, 1 / self.power, out=spec)
            tr.polar(spec, phase, out=self.out_frames_buf)

        rec_audio = tr.istft(self.out_frames_buf,
                             n_fft=self.n_fft,
                             hop_length=self.hop_len,
                             window=self.window,
                             center=self.center,
                             length=self.istft_length)
        rec_audio = rec_audio[:, -self.out_buf_n_samples:]
        if self.fade_n_samples == 0:
            return rec_audio

        self.out_buf[:, -self.fade_n_samples:] *= self.fade_down
        rec_audio[:, :self.fade_n_samples] *= self.fade_up
        rec_audio[:, :self.fade_n_samples] += self.out_buf[:, -self.fade_n_samples:]
        audio_out = rec_audio[:, :self.io_n_samples]
        self.out_buf = rec_audio
        return audio_out

    def _logarithmize_spec(self, spec: Tensor) -> None:
        tr.clamp(spec, min=self.eps, out=spec)
        tr.log10(spec, out=spec)

    def _unlogarithmize_spec(self, spec: Tensor) -> None:
        tr.pow(10, spec, out=spec)
        tr.clamp(spec, min=self.eps, out=spec)

    @tr.jit.export
    def flush(self) -> Tensor:
        # TODO(christhetree): prevent this memory allocation
        audio_out = tr.full((self.batch_size, self.io_n_samples), self.eps)
        if self.fade_n_samples > 0:
            audio_out[:, :self.fade_n_samples] = self.out_buf[:, -self.fade_n_samples:]
        # else:
        #     log.warning('Flushing is not necessary when fade_n_samples == 0')
        return audio_out

    def forward(self,
                audio: Tensor,
                spec_wetdry_ratio: float = 1.0) -> Tensor:
        with tr.no_grad():
            dry_spec = self.audio_to_spec(audio)
            if self.model is None:
                # TODO(christhetree): eliminate memory allocation here
                wet_spec = tr.clone(dry_spec)
            else:
                wet_spec = self.model(dry_spec)

            if self.spec_diff_mode:
                tr.sub(dry_spec, wet_spec, out=wet_spec)

            if spec_wetdry_ratio < 1.0:
                dry_amount = 1.0 - spec_wetdry_ratio
                wet_spec *= spec_wetdry_ratio
                dry_spec *= dry_amount
                wet_spec += dry_spec

            rec_audio = self.spec_to_audio(wet_spec)
            return rec_audio


if __name__ == '__main__':
    batch_size = 1
    io_n_samples = 512
    n_fft = 2048
    hop_len = 512
    # io_n_samples = 2
    # n_fft = 8
    # hop_len = 2
    model_io_n_frames = 16
    power = 1.0
    logarithmize = False
    ensure_pos_spec = True
    # center = True
    center = False
    use_phase_info = True
    fade_n_samples = 0
    audio_n_frames = 19

    rts = RealtimeSTFT(
        batch_size=batch_size,
        io_n_samples=io_n_samples,
        n_fft=n_fft,
        hop_len=hop_len,
        model_io_n_frames=model_io_n_frames,
        center=center,
        power=power,
        logarithmize=logarithmize,
        ensure_pos_spec=ensure_pos_spec,
        use_phase_info=use_phase_info,
        fade_n_samples=fade_n_samples,
    )

    audio = tr.rand((batch_size, hop_len * audio_n_frames))
    assert audio_n_frames % rts.io_n_frames == 0

    # all_spec = rts.audio_to_spec_offline(audio)
    all_spec = tr.stft(audio,
                       n_fft,
                       hop_len,
                       window=rts.window,
                       center=center,
                       return_complex=True)
    all_rec = tr.istft(all_spec,
                       n_fft,
                       hop_len,
                       window=rts.window,
                       center=center,
                       length=audio.shape[1])
    # diff = abs(audio - all_rec)
    # log.info(f'diff.mean() = {diff.mean()}')
    # log.info(f'diff.max() = {diff.max()}')
    # exit()

    all_spec = all_spec.abs()
    chunked_spec = None
    n_steps = (hop_len * audio_n_frames) // io_n_samples
    for idx in range(n_steps):
        start_idx = idx * io_n_samples
        chunk_in = audio[:, start_idx:start_idx + io_n_samples]
        chunked_spec = rts.audio_to_spec(chunk_in)

    all_np = all_spec.numpy()[0]
    chunked_np = chunked_spec.numpy()[0]

    cut_idx = rts.overlap_n_frames - 1
    cut_chunked_np = chunked_np[:, cut_idx:]
    all_np = all_np[:, -model_io_n_frames:]
    cut_all_np = all_np[:, cut_idx:]
    import numpy as np
    print(np.allclose(all_np, chunked_np))
    print(np.allclose(cut_all_np, cut_chunked_np))

    # import matplotlib.pyplot as plt
    # # plt.imshow(all_np)
    # plt.imshow(cut_all_np)
    # plt.title('all_np')
    # plt.show()
    # # plt.imshow(chunked_np)
    # plt.imshow(cut_chunked_np)
    # plt.title('chunked_np')
    # plt.show()
