import logging
import os
from typing import Tuple

from torch import Tensor as T, nn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class SpecCNN(nn.Module):
    def __init__(self,
                 n_filters: int = 1,
                 kernel: Tuple[int, int] = (4,),
                 pooling: Tuple[int, int] = (2,),
                 activation: nn.Module = nn.ELU()) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, n_filters, (3, 3), stride=(1, 1), padding='same', bias=False),
            # activation,
        )

    def forward(self, spec: T) -> T:
        return self.cnn(spec.unsqueeze(1)).squeeze(1)
