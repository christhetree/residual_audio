import logging
import math
import os
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch as tr
from torch import Tensor as T, nn
from torch.optim import Adam
from torchaudio.transforms import Spectrogram

from config import HOP_LENGTH, N_FFT, EPS

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class SpecAE(pl.LightningModule):
    def __init__(self,
                 n_fft: int = N_FFT,
                 hop_length: int = HOP_LENGTH,
                 n_filters: int = 16,
                 kernel: Tuple[int, int] = (4, 4),
                 pooling: Tuple[int, int] = (2, 2),
                 use_bias: bool = True,
                 activation: nn.Module = nn.ELU()) -> None:
        super().__init__()

        self.stft = Spectrogram(n_fft=n_fft,
                                hop_length=hop_length,
                                power=2.0,
                                normalized=False)
        # TODO(christhetree): try using a 2 by 2 stride instead of pooling
        # Set use_bias to false if you add batch normalization
        self.enc = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel, stride=(1, 1), padding='same', bias=use_bias),
            activation,
            nn.MaxPool2d(pooling),
            nn.Conv2d(n_filters, 2 * n_filters, kernel, stride=(1, 1), padding='same', bias=use_bias),
            activation,
            nn.MaxPool2d(pooling),
            nn.Conv2d(2 * n_filters, 4 * n_filters, kernel, stride=(1, 1), padding='same', bias=use_bias),
            activation,
            nn.MaxPool2d(pooling),
            nn.Conv2d(4 * n_filters, 8 * n_filters, kernel, stride=(1, 1), padding='same', bias=use_bias),
            activation,
            nn.MaxPool2d(pooling),
        )
        # I don't know how to use ConvTranspose2d properly so this may be pretty bad
        # TODO(christhetree): fix padding
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(8 * n_filters, 8 * n_filters, (4, 4), stride=pooling, padding=(1, 1), output_padding=(0, 1), bias=use_bias),
            activation,
            nn.ConvTranspose2d(8 * n_filters, 4 * n_filters, (4, 4), stride=pooling, padding=(1, 1), output_padding=(0, 0), bias=use_bias),
            activation,
            nn.ConvTranspose2d(4 * n_filters, 2 * n_filters, (4, 4), stride=pooling, padding=(1, 1), output_padding=(0, 0), bias=use_bias),
            activation,
            nn.ConvTranspose2d(2 * n_filters, n_filters, (4, 4), stride=pooling, padding=(1, 1), output_padding=(1, 0), bias=use_bias),
            activation,
            nn.ConvTranspose2d(n_filters, 1, (1, 1), stride=(1, 1), padding=(0, 0), output_padding=(0, 0)),
            nn.Tanh()
        )

        self.mae = nn.L1Loss(reduction='mean')
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, audio: T) -> (T, ...):
        spec = self.stft(audio).unsqueeze(1)
        spec += EPS
        spec_norm = tr.log10(spec)
        spec_norm /= -math.log10(EPS)
        z = self.enc(spec_norm)
        rec = self.dec(z).squeeze(1)
        spec_norm = spec_norm.squeeze(1)
        return spec_norm, rec

    def _step(self, audio: T, prefix: str) -> T:
        spec_norm, rec = self.forward(audio)

        mae_loss = self.mae(rec, spec_norm)
        mse_loss = self.mse(rec, spec_norm)
        # log_cosh_loss = self.log_cosh(rec, spec_norm)

        loss = mae_loss

        self.log(f'{prefix}_loss',
                 loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        self.log(f'{prefix}_mse',
                 mse_loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True)

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
