import logging
import os
from typing import Tuple

import torch as tr
from torch import Tensor as T, nn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class SpecCNN1D(nn.Module):
    def __init__(self,
                 n_channels: int = 1025,
                 kernel: Tuple[int] = (3,),
                 pooling: Tuple[int] = (2,),
                 activation: nn.Module = nn.ELU()) -> None:
        super().__init__()
        padding = (1,)
        self.enc = nn.Sequential(
            nn.Conv1d(n_channels, n_channels // 2, kernel, stride=pooling, padding=padding),
            activation,
            nn.Conv1d(n_channels // 2, n_channels // 4, kernel, stride=pooling, padding=padding),
            activation,
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(n_channels // 4, n_channels // 2, kernel, stride=pooling, padding=padding, output_padding=(1,)),
            activation,
            nn.ConvTranspose1d(n_channels // 2, n_channels, kernel, stride=pooling, padding=padding, output_padding=(1,)),
            # activation,
        )

    def forward(self, spec: T) -> T:
        z = self.enc(spec)
        rec = self.dec(z)
        return rec


class SpecCNN2D(nn.Module):
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
            # nn.Conv2d(n_filters, n_filters, kernel, stride=pooling, padding=padding),
            # activation,
            # nn.MaxPool2d(kernel, stride=(2, 2), padding=padding),
            nn.Conv2d(n_filters, n_filters * 4, kernel, stride=pooling, padding=padding),
            activation,
            # nn.MaxPool2d(kernel, stride=(2, 2), padding=padding),
            nn.Conv2d(n_filters * 4, n_filters * 16, kernel, stride=pooling, padding=padding),
            activation,
            # nn.MaxPool2d(kernel, stride=(2, 2), padding=padding),
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
