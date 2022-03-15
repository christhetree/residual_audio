import logging
import os
from typing import Tuple

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
            activation,
        )

    def forward(self, spec: T) -> T:
        z = self.enc(spec)
        rec = self.dec(z)
        return rec


class SpecCNN2D(nn.Module):
    def __init__(self,
                 n_filters: int = 16,
                 kernel: Tuple[int] = (3, 3),
                 pooling: Tuple[int] = (2, 2),
                 activation: nn.Module = nn.ELU()) -> None:
        super().__init__()
        padding = (1, 1)
        self.enc = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel, stride=pooling, padding=padding),
            activation,
            nn.Conv2d(n_filters, n_filters * 2, kernel, stride=pooling, padding=padding),
            activation,
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(n_filters * 2, n_filters, kernel, stride=pooling, padding=padding, output_padding=(0, 1)),
            activation,
            nn.ConvTranspose2d(n_filters, n_filters, kernel, stride=pooling, padding=padding, output_padding=(0, 1)),
            activation,
            nn.ConvTranspose2d(n_filters, 1, kernel, stride=(1, 1), padding=padding),
            nn.Tanh(),
        )

    def forward(self, spec: T) -> T:
        spec = spec.unsqueeze(1)
        z = self.enc(spec)
        rec = self.dec(z)
        rec = rec.squeeze(1)
        return rec
