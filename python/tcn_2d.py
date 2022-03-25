"""
Based off
https://github.com/csteinmetz1/steerable-nafx/blob/main/steerable-nafx.ipynb
"""
import logging
import os
from typing import Tuple, Optional

from torch import Tensor
import torch as tr
from torch import nn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def causal_crop(x: Tensor, length: int) -> Tensor:
    if x.shape[-1] != length:
        stop = x.shape[-1] - 1
        start = stop - length
        x = x[..., start:stop]
    return x


class FiLM(nn.Module):
    def __init__(
            self,
            cond_dim: int,  # dim of conditioning input
            num_features: int,  # dim of the conv channel
            batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.adaptor = nn.Linear(cond_dim, num_features * 2)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        cond = self.adaptor(cond)
        g, b = tr.chunk(cond, 2, dim=-1)
        g = g.permute(0, 2, 1)
        b = b.permute(0, 2, 1)

        if self.batch_norm:
            x = self.bn(x)  # apply BatchNorm without affine
        x = (x * g) + b  # then apply conditional affine

        return x


class TCN2DBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 kernel: Tuple[int],
                 dilation: Tuple[int],
                 cond_dim: int = 0,
                 activation: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel,
            dilation=dilation,
            padding=0,  # ((kernel_size-1)//2)*dilation,
            bias=True,
        )
        if cond_dim > 0:
            self.film = FiLM(cond_dim, out_ch, batch_norm=False)
        if activation:
            self.act = nn.PReLU()
        self.res = nn.Conv1d(in_ch, out_ch, (1,), bias=False)

    def forward(self, x: Tensor, c: Optional[Tensor] = None) -> Tensor:
        x_in = x
        x = self.conv(x)
        if hasattr(self, "film"):
            x = self.film(x, c)
        if hasattr(self, "act"):
            x = self.act(x)
        x_res = causal_crop(self.res(x_in), x.shape[-1])
        x = x + x_res

        return x


class TCN2D(nn.Module):
    def __init__(self,
                 n_inputs: int = 1,
                 n_outputs: int = 1,
                 n_blocks: int = 10,
                 kernel_size: int = 13,
                 n_channels: int = 64,
                 dilation_growth: int = 4,
                 cond_dim: int = 0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.n_channels = n_channels
        self.dilation_growth = dilation_growth
        self.n_blocks = n_blocks
        self.stack_size = n_blocks

        self.blocks = nn.ModuleList()
        for n in range(n_blocks):
            if n == 0:
                in_ch = n_inputs
                out_ch = n_channels
            elif (n + 1) == n_blocks:
                in_ch = n_channels
                out_ch = n_outputs
            else:
                in_ch = n_channels
                out_ch = n_channels

            dilation = dilation_growth ** n
            self.blocks.append(TCN2DBlock(
                in_ch,
                out_ch,
                (kernel_size,),
                (dilation,),
                cond_dim=cond_dim,
                activation=True
            ))

    def forward(self, x: Tensor, c: Tensor = None) -> Tensor:
        for block in self.blocks:
            x = block(x, c)
        return x

    def compute_receptive_field(self) -> int:
        """Compute the receptive field in samples."""
        rf = self.kernel_size
        for n in range(1, self.n_blocks):
            dilation = self.dilation_growth ** (n % self.stack_size)
            rf = rf + ((self.kernel_size - 1) * dilation)
        return rf


if __name__ == '__main__':
    tcn = TCN2D(n_blocks=4)
    audio = tr.rand((1, 65536))
    out = tcn.forward(audio)
    log.info(out.shape)
