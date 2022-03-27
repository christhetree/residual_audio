import logging
import os
from typing import Optional

import torch as tr
from torch import Tensor
from torch import nn
from torchaudio.transforms import Spectrogram, InverseSpectrogram

from config import EPS, MODEL_IO_N_FRAMES
from modeling import SpecCNN2DSmall

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
                 model_io_n_frames: int = MODEL_IO_N_FRAMES,
                 spec_diff_mode: bool = False,
                 power: Optional[float] = 1.0,
                 logarithmize: bool = True,
                 use_phase_info: bool = True,
                 fade_n_samples: int = 0,
                 eps: float = EPS) -> None:
        super().__init__()
        assert io_n_samples >= hop_len
        assert io_n_samples % hop_len == 0
        assert n_fft % 2 == 0
        assert (n_fft // 2) % hop_len == 0
        assert power is None or power >= 1.0
        if power > 1.0:
            log.warning('A power greater than 1.0 probably adds unnecessary '
                        'computational complexity')
        assert fade_n_samples <= io_n_samples
        self.model = model
        self.batch_size = batch_size
        self.io_n_samples = io_n_samples
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.model_io_n_frames = model_io_n_frames
        self.spec_diff_mode = spec_diff_mode
        self.power = power
        self.logarithmize = logarithmize
        self.use_phase_info = use_phase_info
        self.fade_n_samples = fade_n_samples
        self.eps = eps

        self.io_n_frames = self.io_n_samples // self.hop_len
        self.overlap_n_frames = self.n_fft // 2 // self.hop_len
        self.in_buf_n_frames = (self.overlap_n_frames * 2) - 1 + self.io_n_frames
        self.n_bins = (self.n_fft // 2) + 1
        self.stft_out_shape = (self.batch_size, self.n_bins, self.in_buf_n_frames + 1)
        self.model_io_shape = (self.batch_size, self.n_bins, self.model_io_n_frames)
        self.out_buf_n_samples = self.io_n_samples + self.fade_n_samples
        assert self.out_buf_n_samples <= (self.in_buf_n_frames - 1) * self.hop_len

        # TODO(christhetree): implement center=False case
        self.stft = Spectrogram(n_fft=self.n_fft,
                                hop_length=self.hop_len,
                                pad=0,
                                center=True,
                                normalized=False,
                                power=None)
        self.istft = InverseSpectrogram(n_fft=self.n_fft,
                                        hop_length=self.hop_len,
                                        pad=0,
                                        center=True,
                                        normalized=False)

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
            (self.batch_size, self.n_bins, self.in_buf_n_frames),
            self.eps,
            dtype=tr.complex64,
        )
        self.out_buf = tr.full(
            (self.batch_size, self.out_buf_n_samples),
            self.eps,
        )
        self.reset()

        # These must be instantiated for TorchScript
        self.fade_up = tr.linspace(0, 1, max(self.fade_n_samples, 1))
        self.fade_down = tr.linspace(1, 0, max(self.fade_n_samples, 1))
        self.zero_phase = tr.zeros(self.model_io_shape)

    @tr.jit.export
    def calc_min_delay_samples(self) -> int:
        return self.fade_n_samples

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
        # Remove overlap frames we have computed before
        frames = stft_out_buf[:, :, self.overlap_n_frames:]
        # Identify frames that are more correct due to missing prev audio info
        fixed_prev_frames = frames[:, :, :-self.io_n_frames]
        assert fixed_prev_frames.shape[2] == self.overlap_n_frames
        # Identify the new frames for the input audio chunk
        new_frames = frames[:, :, -self.io_n_frames:]
        # Overwrite previous frames with more correct frames
        frames_buf[:, :, -self.overlap_n_frames:] = fixed_prev_frames
        # Shift buffer left and insert new frames
        frames_buf = tr.roll(frames_buf, -self.io_n_frames, dims=2)
        frames_buf[:, :, -self.io_n_frames:] = new_frames
        return frames_buf

    @tr.jit.ignore
    def audio_to_spec_offline(self, audio: Tensor) -> Tensor:
        assert audio.shape[0] == self.batch_size
        assert audio.shape[1] >= self.n_fft
        assert audio.shape[1] % self.hop_len == 0
        spec = self.stft(audio)
        if self.power is None:
            spec = spec.real
        else:
            spec = spec.abs()
            if self.power != 1.0:
                spec = spec.pow(self.power)

        if self.logarithmize:
            spec = tr.clamp(spec, min=self.eps)
            spec = tr.log10(spec)

        return spec

    @tr.jit.export
    def audio_to_spec(self, audio: Tensor) -> Tensor:
        assert audio.shape == (self.batch_size, self.io_n_samples)
        # Shift buffer left and insert audio chunk
        self.in_buf = tr.roll(self.in_buf, -self.io_n_samples, dims=1)
        self.in_buf[:, -self.io_n_samples:] = audio

        complex_frames = self.stft(self.in_buf)
        if self.power is None:
            self.stft_mag_buf = complex_frames.real
        else:
            tr.abs(complex_frames, out=self.stft_mag_buf)
            if self.power != 1.0:
                tr.pow(self.stft_mag_buf, self.power, out=self.stft_mag_buf)
        if self.logarithmize:
            self._logarithmize_spec(self.stft_mag_buf)

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
        spec = spec[:, :, -self.in_buf_n_frames:]
        if self.use_phase_info:
            phase = self.phase_buf[:, :, -self.in_buf_n_frames:]
        else:
            phase = self.zero_phase[:, :, -self.in_buf_n_frames:]

        if self.logarithmize:
            self._unlogarithmize_spec(spec)

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

    def forward(self, audio: Tensor, spec_wetdry_ratio: float = 1.0) -> Tensor:
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
    # TODO(christhetree): fix io_n_samples = 1024, model_io_n_frames = 4 bug
    # model = None
    model = SpecCNN2DSmall()
    # batch_size = 1
    # hop_length = 512
    # io_n_samples = 512
    # n_fft = 2048
    # model_io_n_frames = 16
    # fade_n_samples = 0
    # power = 1.0
    # logarithmize = True
    # use_phase_info = True
    # audio_n_frames = 16
    #
    # rts = RealtimeSTFT(
    #     model,
    #     batch_size,
    #     io_n_samples,
    #     n_fft,
    #     hop_length,
    #     model_io_n_frames,
    #     power,
    #     logarithmize,
    #     use_phase_info,
    #     fade_n_samples,
    # )

    # audio = tr.rand((batch_size, hop_length * audio_n_frames))
    # assert audio_n_frames % rts.io_n_frames == 0
    #
    # all_spec = rts.stft(audio)
    # all_spec = all_spec.abs().pow(power)
    #
    # chunked_spec = None
    # n_steps = (hop_length * audio_n_frames) // io_n_samples
    # for idx in range(n_steps):
    #     start_idx = idx * io_n_samples
    #     chunk_in = audio[:, start_idx:start_idx + io_n_samples]
    #     chunked_spec = rts.audio_to_spec(chunk_in)
    #     # print(np.allclose(chunk_in.numpy(), chunk_out.numpy()))
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
