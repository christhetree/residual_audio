import logging
import os
from typing import Tuple

import torch as tr
from torch import Tensor as T, nn

from config import N_FFT, N_BINS
from tcn_2d import TCN2D

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class SpecCNN1D(nn.Module):
    def __init__(self,
                 n_channels: int = (N_FFT // 2) + 1,
                 n_filters: int = 4,
                 kernel: Tuple[int] = (5,),
                 pooling: Tuple[int] = (2,),
                 activation: nn.Module = nn.ELU()) -> None:
        super().__init__()
        padding = (kernel[0] // 2,)
        self.enc = nn.Sequential(
            nn.Conv1d(n_channels, n_filters, kernel, stride=pooling, padding=padding),
            activation,
            nn.Conv1d(n_filters, n_filters, kernel, stride=pooling, padding=padding),
            activation,
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(n_filters, n_filters, kernel, stride=pooling, padding=padding, output_padding=(1,)),
            activation,
            nn.ConvTranspose1d(n_filters, n_filters, kernel, stride=pooling, padding=padding, output_padding=(1,)),
            activation,
            nn.ConvTranspose1d(n_filters, n_channels, kernel, stride=(1,), padding=padding),
        )

    def forward(self, spec: T) -> T:
        z = self.enc(spec)
        rec = self.dec(z)
        return rec


class SpecCNN2DSmall(nn.Module):
    def __init__(self,
                 n_filters: int = 4,
                 kernel: Tuple[int] = (5, 5),
                 pooling: Tuple[int] = (4, 2),
                 activation: nn.Module = nn.ELU()) -> None:
        super().__init__()
        padding = (kernel[0] // 2, kernel[1] // 2)
        self.enc = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel, stride=pooling, padding=padding),
            activation,
            nn.Conv2d(n_filters, n_filters * 4, kernel, stride=pooling, padding=padding),
            activation,
            nn.Conv2d(n_filters * 4, n_filters * 16, kernel, stride=pooling, padding=padding),
            activation,
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(n_filters * 16, n_filters * 4, kernel, stride=pooling, padding=padding, output_padding=(0, 1)),
            activation,
            nn.ConvTranspose2d(n_filters * 4, n_filters, kernel, stride=pooling, padding=padding, output_padding=(0, 1)),
            activation,
            nn.ConvTranspose2d(n_filters, n_filters, kernel, stride=pooling, padding=padding, output_padding=(0, 1)),
            activation,
            nn.ConvTranspose2d(n_filters, 1, kernel, stride=(1, 1), padding=padding),
        )

    def forward(self, spec: T) -> T:
        spec = spec.unsqueeze(1)
        z = self.enc(spec)
        rec = self.dec(z)
        rec = rec.squeeze(1)
        return rec


class SpecCNN2DLarge(SpecCNN2DSmall):
    def __init__(self,
                 n_filters: int = 16,
                 kernel: Tuple[int] = (5, 3),
                 pooling: Tuple[int] = (4, 1),
                 activation: nn.Module = nn.LeakyReLU(0.1)) -> None:
        super().__init__()
        padding = (kernel[0] // 2, kernel[1] // 2)
        self.enc = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel, stride=pooling, padding=padding),
            activation,
            nn.BatchNorm2d(n_filters),
            nn.Conv2d(n_filters, 2 * n_filters, kernel, stride=(2, 2), padding=padding),
            activation,
            nn.BatchNorm2d(2 * n_filters),
            nn.Conv2d(2 * n_filters, 4 * n_filters, kernel, stride=pooling, padding=padding),
            activation,
            nn.BatchNorm2d(4 * n_filters),
            nn.Conv2d(4 * n_filters, 8 * n_filters, kernel, stride=(2, 2), padding=padding),
            activation,
            nn.BatchNorm2d(8 * n_filters),
            nn.Conv2d(8 * n_filters, 16 * n_filters, kernel, stride=(2, 2), padding=padding),
            activation,
            nn.BatchNorm2d(16 * n_filters),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(16 * n_filters, 8 * n_filters, kernel, stride=(2, 2), padding=padding, output_padding=(0, 1)),
            activation,
            nn.BatchNorm2d(8 * n_filters),
            nn.ConvTranspose2d(8 * n_filters, 4 * n_filters, kernel, stride=(2, 2), padding=padding, output_padding=(0, 1)),
            activation,
            nn.BatchNorm2d(4 * n_filters),
            nn.ConvTranspose2d(4 * n_filters, 2 * n_filters, kernel, stride=pooling, padding=padding),
            activation,
            nn.BatchNorm2d(2 * n_filters),
            nn.ConvTranspose2d(2 * n_filters, n_filters, kernel, stride=(2, 2), padding=padding, output_padding=(0, 1)),
            activation,
            nn.BatchNorm2d(n_filters),
            nn.ConvTranspose2d(n_filters, n_filters, kernel, stride=pooling, padding=padding),
            activation,
            nn.BatchNorm2d(n_filters),
            nn.ConvTranspose2d(n_filters, 1, kernel, stride=(1, 1), padding=padding),
        )


class SpecCNN2DMinimal(SpecCNN2DSmall):
    def __init__(self,
                 n_filters: int = 8,
                 kernel: Tuple[int] = (5, 5),
                 pooling: Tuple[int] = (2, 1),
                 activation: nn.Module = nn.ELU()) -> None:
        super().__init__()
        padding = (kernel[0] // 2, kernel[1] // 2)
        self.enc = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel, stride=pooling, padding=padding),
            activation,
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(n_filters, n_filters, kernel, stride=pooling, padding=padding),
            activation,
            nn.ConvTranspose2d(n_filters, 1, kernel, stride=(1, 1), padding=padding),
        )


class SpecMLP(nn.Module):
    def __init__(self,
                 n_channels: int = 1025,
                 activation: nn.Module = nn.ELU()) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_channels, n_channels // 2),
            activation,
        )
        self.dec = nn.Sequential(
            nn.Linear(n_channels // 2, n_channels),
        )

    def forward(self, spec: T) -> T:
        spec = tr.swapaxes(spec, 1, 2)
        z = self.enc(spec)
        rec = self.dec(z)
        rec = tr.swapaxes(rec, 1, 2)
        return rec


class FXModel(nn.Module):
    def __init__(self,
                 n_filters: int = 1,
                 kernel: Tuple[int] = (5, 5)) -> None:
        super().__init__()
        self.chain = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel, padding='same', bias=True),
            # nn.Linear(N_BINS, N_BINS),
            # TCN2D(n_blocks=2, n_channels=1, use_act=True, use_bn=False),
        )

    def forward(self, spec: T) -> T:
        spec = spec.unsqueeze(1)
        # spec = tr.swapaxes(spec, 1, 2)
        wet_spec = self.chain(spec)
        wet_spec = wet_spec.squeeze(1)
        # wet_spec = tr.swapaxes(wet_spec, 1, 2)
        return wet_spec
