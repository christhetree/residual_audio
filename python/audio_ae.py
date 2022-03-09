import logging
import math
import os
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch as tr
from torch import Tensor as T, nn
from torch.optim import Adam
from torchaudio.transforms import Spectrogram, GriffinLim, InverseSpectrogram

from config import HOP_LENGTH, N_FFT, EPS, N_SAMPLES, PEAK_VALUE

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class SpecAE(pl.LightningModule):
    def __init__(self,
                 n_fft: int = N_FFT,
                 hop_length: int = HOP_LENGTH,
                 n_filters: int = 1,
                 kernel: Tuple[int, int] = (4,),
                 pooling: Tuple[int, int] = (2,),
                 activation: nn.Module = nn.ELU(),
                 eps: float = EPS) -> None:
        super().__init__()

        self.stft = Spectrogram(n_fft=n_fft,
                                hop_length=hop_length,
                                power=2.0,
                                normalized=False)

        n_channels = (n_fft // 2) + 1
        self.enc = nn.Sequential(
            nn.Conv1d(n_channels, n_filters, kernel, stride=pooling, padding=0),
            activation,
            nn.Conv1d(n_filters, 2 * n_filters, kernel, stride=pooling, padding=0),
            activation,
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(2 * n_filters, n_filters, kernel, stride=pooling, output_padding=(1,)),
            activation,
            nn.ConvTranspose1d(n_filters, n_channels, kernel, stride=pooling, output_padding=(1,)),
            activation,
            nn.ConvTranspose1d(n_channels, n_channels, (1,), stride=(1,)),
            nn.Tanh(),
        )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.eps = eps
        self.mae = nn.L1Loss(reduction='mean')
        # self.mse = nn.MSELoss(reduction='mean')

    def forward(self, audio: T) -> Tuple[T, T]:
        spec = self.stft(audio)
        spec += self.eps
        spec_norm = tr.log10(spec)
        spec_norm /= -math.log10(self.eps)
        z = self.enc(spec_norm)
        rec = self.dec(z)
        return spec_norm, rec

    def _step(self, audio: T, prefix: str) -> T:
        spec_norm, rec_norm = self.forward(audio)

        # spec = unnormalize_spec(spec_norm)
        # rec = unnormalize_spec(rec_norm)
        # diff = calc_diff_spec(spec, rec)
        # diff_mean = diff.mean(dim=[1, 2])
        # diff_std = diff.std(dim=[1, 2])
        # diff_snr = diff_mean / diff_std
        # snr_loss = 1 / tr.mean(diff_snr)
        # snr_loss /= 100.0

        mae_loss = self.mae(rec_norm, spec_norm)
        # mse_loss = self.mse(rec_norm, spec_norm)

        loss = mae_loss
        # loss = mae_loss + snr_loss

        self.log(f'{prefix}_loss',
                 loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        # self.log(f'{prefix}_mae',
        #          mae_loss,
        #          on_step=True,
        #          on_epoch=True,
        #          prog_bar=True)
        # self.log(f'{prefix}_snr',
        #          snr_loss,
        #          on_step=True,
        #          on_epoch=True,
        #          prog_bar=True)

        return loss

    def training_step(self,
                      batch: T,
                      batch_idx: Optional[int] = None) -> T:
        return self._step(batch, 'train')

    def validation_step(self,
                        batch: T,
                        batch_idx: Optional[int] = None) -> T:
        return self._step(batch, 'val')

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer


class ResidualAudioEffect(pl.LightningModule):
    def __init__(self,
                 spec_ae: SpecAE,
                 buffer_size: int = 2048,
                 n_samples: int = N_SAMPLES,
                 peak_value: float = PEAK_VALUE) -> None:
        super().__init__()
        self.spec_ae = spec_ae
        self.buffer_size = buffer_size
        self.n_samples = n_samples
        self.peak_value = peak_value
        self.eps = spec_ae.eps
        self.gl = GriffinLim(n_fft=spec_ae.n_fft,
                             hop_length=spec_ae.hop_length,
                             power=2.0)
        self.istft = InverseSpectrogram(n_fft=spec_ae.n_fft,
                                        hop_length=spec_ae.hop_length,
                                        normalized=False)

    def unnormalize_spec(self, spec_norm: T) -> T:
        spec = (10 ** (spec_norm * (-math.log10(self.eps)))) - self.eps
        spec = tr.clamp(spec, 0.0)
        return spec

    def calc_diff_spec(self, spec: T, rec: T, alpha: T) -> T:
        spec_max = tr.amax(spec, dim=[1, 2], keepdim=True)
        spec = spec / spec_max

        rec_max = tr.amax(rec, dim=[1, 2], keepdim=True)
        rec = rec / rec_max

        diff = tr.clamp(spec - (alpha * rec), 0.0)
        diff *= tr.maximum(spec_max, rec_max)

        return diff

    # def normalize_waveform(self, x: T) -> T:
    #     assert len(x.shape) == 1
    #     return (x / max(abs(x.max()), abs(x.min()))) * self.peak_value

    def forward(self, audio: T, alpha: T) -> T:
        assert audio.shape == (self.buffer_size,)
        ae_in = tr.zeros(1, self.n_samples)
        ae_in[:, :self.buffer_size] = audio
        spec_norm, rec_norm = self.spec_ae.forward(ae_in)
        spec = self.unnormalize_spec(spec_norm)
        rec = self.unnormalize_spec(rec_norm)
        diff = self.calc_diff_spec(spec, rec, alpha)
        diff_gl = self.gl(diff).squeeze(0)
        # diff_gl = self.istft(diff).squeeze(0)
        diff_gl = diff_gl[:self.buffer_size]
        # diff_gl_norm = self.normalize_waveform(diff_gl)
        # return diff_gl_norm
        return diff_gl
