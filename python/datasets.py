import logging
import os
from typing import List, Optional

import torch as tr
from torch import Tensor as T
from torch.utils.data import Dataset

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class PathsDataset(Dataset):
    def __init__(self,
                 paths: List[str],
                 fx_dir: Optional[str] = None) -> None:
        self.paths = paths
        self.fx_paths = None
        if fx_dir is not None:
            self.fx_paths = []
            for path in self.paths:
                name = os.path.basename(path)
                fx_path = os.path.join(fx_dir, name)
                self.fx_paths.append(fx_path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> (T, T):
        path = self.paths[idx]
        t_in = tr.load(path)
        if self.fx_paths is not None:
            fx_path = self.fx_paths[idx]
            t_out = tr.load(fx_path)
        else:
            t_out = tr.clone(t_in)

        return t_in, t_out
