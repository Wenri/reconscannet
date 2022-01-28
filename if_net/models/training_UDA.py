import random

import torch
from torch.nn import functional as F

from net_utils.voxel_util import pointcloud2voxel_fast
from .training import Trainer
from ..data_processing import implicit_waterproofing as iw


class TrainerUDA(Trainer):
    def __init__(self, model, model_tea, device, exp_name, optimizer, warmup_iters, warmup_factor=1. / 1000,
                 balance_weight=False):
        self._rng = random.SystemRandom()
        for param in model_tea.parameters():
            param.detach_()
        self.model_tea = model_tea.to(device=device)

        self.min = -0.5
        self.max = 0.5
        self.resolution = 24

        grid_points = iw.create_grid_points_from_bounds(self.min, self.max, self.resolution)
        grid_points[:, 0], grid_points[:, 2] = grid_points[:, 2], grid_points[:, 0].copy()

        a = self.max + self.min
        b = self.max - self.min

        grid_coords = 2 * grid_points - a
        grid_coords = grid_coords / b

        grid_coords = torch.from_numpy(grid_coords).to(device, dtype=torch.float)
        self.grid_coords = torch.reshape(grid_coords, (1, len(grid_points), 3)).to(device)

        super(TrainerUDA, self).__init__(model, device, exp_name, optimizer, warmup_iters,
                                         warmup_factor=warmup_factor, balance_weight=balance_weight)

    def update_ema_variables(self, alpha, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(self.model_tea.parameters(), self.model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def consistency(self, partial, cls_codes=None):
        n_partial = partial.shape[0]
        partial_input, p = self.overscan_aug(partial, self.grid_coords)

        voxel_grids = pointcloud2voxel_fast(partial_input)
        # self.visualize('debug', voxel_grids, partial_input, p, occ)

        input_features = self.model.infer_c(partial_input.transpose(1, 2), cls_codes_for_completion=cls_codes)
        input_features_tea = self.model_tea.infer_c(partial_input.transpose(1, 2), cls_codes_for_completion=cls_codes)

        z = self.model.get_z_from_prior((n_partial,), sample=True, device=self.device)
        z_tea = self.model_tea.get_z_from_prior((n_partial,), sample=True, device=self.device)

        ret = self.model.decode(p, z, input_features, voxel_grids).logits
        ret_tea = self.model_tea.decode(p, z_tea, input_features_tea, voxel_grids).logits

        loss = F.mse_loss(F.sigmoid(ret), F.sigmoid(ret_tea), reduction='sum')
        return loss

    def compute_loss(self, partial, valid_mask, p, occ, cls_codes=None, abnormal=None, **kwargs):
        loss, ret = super(TrainerUDA, self).compute_loss(partial, valid_mask, p, occ, cls_codes=cls_codes, **kwargs)
        ab_partial = abnormal['partial_pc'].to(device=self.device)
        loss += self.consistency(ab_partial, cls_codes=abnormal['cls'])
        return loss, ret
