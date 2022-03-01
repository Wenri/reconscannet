import random

import torch

from .training_UDA import TrainerUDA


class TrainerABNormal(TrainerUDA):
    def __init__(self, model, model_tea, device, exp_name, optimizer, warmup_iters, warmup_factor=1. / 1000,
                 balance_weight=False):
        self._rng = random.SystemRandom()
        super(TrainerABNormal, self).__init__(model, model_tea, device, exp_name, optimizer, warmup_iters,
                                              warmup_factor=warmup_factor, balance_weight=balance_weight)

    def compute_loss(self, partial, valid_mask, p, occ, cls_codes=None, abnormal=None, **kwargs):
        n_in = abnormal['n_in'].to(dtype=torch.bool, device=self.device)
        ab_pts = abnormal['pts'].to(device=self.device)
        ab_occ = abnormal['pts_mask'].to(device=self.device).float()
        ab_partial = abnormal['partial_pc'].to(device=self.device)
        n_abnormal = torch.count_nonzero(n_in)
        batch_size = partial.shape[0]
        replace_id = set(torch.logical_not(valid_mask).nonzero(as_tuple=True)[0][:n_abnormal].tolist())
        while len(replace_id) < n_abnormal:
            replace_id.add(self._rng.randrange(batch_size))
        replace_id = list(replace_id)
        valid_mask[replace_id] = 1.
        partial[replace_id] = ab_partial[n_in]
        p[replace_id] = ab_pts[n_in]
        occ[replace_id] = ab_occ[n_in]
        cls_codes[replace_id] = abnormal['cls'][n_in]
        return super(TrainerABNormal, self).compute_loss(
            partial, valid_mask, p, occ, cls_codes=cls_codes, abnormal=abnormal, **kwargs)
