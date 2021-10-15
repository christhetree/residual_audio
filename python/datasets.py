import logging
import os
from typing import List

import torch as tr
from torch import Tensor as T
from torch.utils.data import Dataset

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class PathsDataset(Dataset):
    def __init__(self, paths: List[str]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> T:
        path = self.paths[idx]
        t = tr.load(path)
        return t
