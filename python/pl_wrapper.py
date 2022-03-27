import logging
import os
from typing import Optional, List, Any, Dict

import torch as tr
from pytorch_lightning import LightningModule
from torch import Tensor as T, nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import MultiplicativeLR

from realtime_stft import RealtimeSTFT

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class PLWrapper(LightningModule):
    def __init__(self,
                 model: nn.Module,
                 rts: RealtimeSTFT,
                 init_lr: float = 1e-3) -> None:
        super().__init__()
        self.model = model
        self.rts = rts
        self.init_lr = init_lr
        self.mae = nn.L1Loss(reduction='mean')

    def forward(self, batch: List[T]) -> (T, T):
        audio_in = batch[0]
        audio_out = batch[1]
        spec_in = self.rts.audio_to_spec_offline(audio_in).detach()
        rec = self.model(spec_in)
        spec_out = spec_in
        if not tr.allclose(audio_in, audio_out):  # TODO(christhetree)
            spec_out = self.rts.audio_to_spec_offline(audio_out).detach()
        return spec_out, rec

    def _step(self, batch: List[T], prefix: str) -> T:
        spec_out, rec = self.forward(batch)
        mae_loss = self.mae(rec, spec_out)
        loss = mae_loss
        self.log(f'{prefix}_loss', loss, prog_bar=True)
        return loss

    def training_step(self,
                      batch: List[T],
                      batch_idx: Optional[int] = None) -> T:
        return self._step(batch, 'train')

    def validation_step(self,
                        batch: List[T],
                        batch_idx: Optional[int] = None) -> T:
        return self._step(batch, 'val')

    def configure_optimizers(self) -> (List[Optimizer], List[Dict[str, Any]]):
        opt = Adam(self.parameters(), lr=self.init_lr)
        lr_scheduler_1 = MultiplicativeLR(opt, lr_lambda=lambda x: 0.90)
        lr_schedule = {
            'scheduler': lr_scheduler_1,
            'monitor': 'val_loss',
        }
        return [opt], [lr_schedule]
