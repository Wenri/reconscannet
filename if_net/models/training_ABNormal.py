import random

import torch

from .training import Trainer


class TrainerABNormal(Trainer):
    def __init__(self, model, device, exp_name, optimizer, warmup_iters, warmup_factor=1. / 1000, balance_weight=False):
        self._rng = random.SystemRandom()
        super(TrainerABNormal, self).__init__(model, device, exp_name, optimizer, warmup_iters,
                                              warmup_factor=warmup_factor, balance_weight=balance_weight)

    def compute_loss(self, partial, valid_mask, p, occ, cls_codes=None, abnormal=None, **kwargs):
        ab_pts = abnormal['pts'].to(device=self.device)
        ab_occ = abnormal['pts_mask'].to(device=self.device).float()
        ab_partial = abnormal['partial_pc'].to(device=self.device)
        n_abnormal = ab_pts.shape[0]
        batch_size = partial.shape[0]
        replace_id = set(torch.logical_not(valid_mask).nonzero(as_tuple=True)[0][:n_abnormal].tolist())
        while len(replace_id) < n_abnormal:
            replace_id.add(self._rng.randrange(batch_size))
        replace_id = list(replace_id)
        valid_mask[replace_id] = 1.
        partial[replace_id] = ab_partial
        p[replace_id] = ab_pts
        occ[replace_id] = ab_occ
        return super(TrainerABNormal, self).compute_loss(partial, valid_mask, p, occ, cls_codes=cls_codes, **kwargs)
