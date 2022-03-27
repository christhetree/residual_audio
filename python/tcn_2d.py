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

from tcn_1d import causal_crop

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class FiLM(nn.Module):
    def __init__(
            self,
            cond_dim: int,  # dim of conditioning input
            num_features: int,  # dim of the conv channel
            use_bn: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm2d(num_features, affine=False)
        # TODO(christhetree): add dynamic layers
        self.adaptor = nn.Linear(cond_dim, 2 * num_features)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        cond = self.adaptor(cond)
        g, b = tr.chunk(cond, 2, dim=-1)
        g = g[:, :, None, None]
        b = b[:, :, None, None]

        if self.use_bn:
            x = self.bn(x)  # Apply batchnorm without affine
        x = (x * g) + b  # Then apply conditional affine

        return x


class TCN2DBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 kernel_shape: Tuple[int, int],
                 dilation: Tuple[int, int],
                 bin_padding: Optional[int] = None,
                 cond_dim: int = 0,
                 use_act: bool = True,
                 use_bn: bool = True) -> None:
        super().__init__()
        self.bin_padding = bin_padding
        if self.bin_padding is None:
            self.bin_padding = ((kernel_shape[0] - 1) // 2) * dilation[0]

        self.act = None
        if use_act:
            self.act = nn.PReLU()

        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_shape,
            dilation=dilation,
            padding=(self.bin_padding, 0),
            bias=True,
        )
        self.res = nn.Conv2d(in_ch, out_ch, (1, 1), bias=False)

        self.film = None
        if cond_dim > 0:
            self.film = FiLM(cond_dim, out_ch, use_bn=use_bn)

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        x_in = x
        x = self.conv(x)
        if self.film is not None:
            x = self.film(x, cond)
        if self.act is not None:
            x = self.act(x)

        res = self.res(x_in)
        x_res = causal_crop(res, x.shape[-1])
        x += x_res

        return x


class TCN2D(nn.Module):
    def __init__(self,
                 in_ch: int = 1,
                 out_ch: int = 1,
                 n_blocks: int = 4,
                 kernel_shape: Tuple[int, int] = (5, 5),
                 n_channels: int = 32,
                 bins_dil_growth: int = 2,
                 time_dil_growth: int = 2,
                 bin_padding: Optional[int] = None,
                 cond_dim: int = 0,
                 use_act: bool = True,
                 use_bn: bool = False) -> None:
        super().__init__()
        self.kernel_shape = kernel_shape
        self.n_channels = n_channels
        self.bins_dil_growth = bins_dil_growth
        self.time_dil_growth = time_dil_growth
        self.n_blocks = n_blocks
        self.stack_size = n_blocks

        self.blocks = nn.ModuleList()
        for n in range(n_blocks):
            if n == 0:
                block_in_ch = in_ch
                block_out_ch = self.n_channels
            elif n == n_blocks - 1:
                block_in_ch = self.n_channels
                block_out_ch = out_ch
            else:
                block_in_ch = self.n_channels
                block_out_ch = self.n_channels

            bins_dil = self.bins_dil_growth ** n
            time_dil = self.time_dil_growth ** n
            self.blocks.append(TCN2DBlock(
                block_in_ch,
                block_out_ch,
                self.kernel_shape,
                (bins_dil, time_dil),
                bin_padding=bin_padding,
                cond_dim=cond_dim,
                use_act=use_act,
                use_bn=use_bn,
            ))

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        assert len(x.shape) == 4  # (batch_size, in_ch, bins, frames)
        if cond is not None:
            assert len(cond.shape) == 2  # (batch_size, cond_dim)
        for block in self.blocks:
            x = block(x, cond)
        return x

    def calc_receptive_field(self) -> int:
        """Compute the receptive field in frames."""
        rf = self.kernel_shape[-1]
        for idx in range(1, self.n_blocks):
            dilation = self.time_dil_growth ** (idx % self.stack_size)
            rf = rf + ((self.kernel_shape[-1] - 1) * dilation)
        return rf


if __name__ == '__main__':
    tcn = TCN2D(n_blocks=1, cond_dim=3, use_bn=True)
    log.info(tcn.calc_receptive_field())
    spec = tr.rand((1, 1, 1025, 32))
    cond = tr.rand((1, 3))
    out = tcn.forward(spec, cond)
    log.info(out.shape)
