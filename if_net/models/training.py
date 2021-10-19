from __future__ import division

import os
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from export_scannet_pts import vox_to_mesh, write_pointcloud
from net_utils.voxel_util import pointcloud2voxel_fast


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


class Trainer(object):
    def __init__(self, model, device, exp_name, optimizer, warmup_iters, warmup_factor=1. / 1000, balance_weight=False):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.exp_path = os.path.dirname(__file__) + '/../experiments/{}/'.format(exp_name)
        self.checkpoint_path = self.exp_path + 'checkpoints/'.format(exp_name)
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.writer = SummaryWriter(self.exp_path + 'summary'.format(exp_name))
        self.val_min = None
        self.balance_weight = balance_weight
        self.lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss, _ = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return loss.item()

    def compute_loss(self, batch):
        device = self.device
        overscan_factor = 0.1

        partial = batch.get('partial').to(device)
        p = batch.get('points').to(device)
        occ = batch.get('points.occ').to(device)
        # voxel_gt = batch.get('voxels').to(device)
        partial_aug = batch.get('partial_aug').to(device)
        partial_aug_valid = batch.get('partial_aug.valid').unsqueeze(-1).unsqueeze(-1).to(device)
        partial_input = torch.where(partial_aug_valid, partial_aug, partial)
        n_batch = p.shape[0]

        voxel_grids = pointcloud2voxel_fast(partial_input)
        overscan = torch.abs(torch.randn(*partial_aug_valid.shape, device=device)) * overscan_factor
        offset = (torch.rand(n_batch, 1, 3, device=device) - 0.5) * overscan

        partial_input = (partial_input + offset) / (overscan + 1)
        p = (p + offset) / (overscan + 1)

        cls_codes = batch.get('category')
        if cls_codes is not None:
            cls_codes = F.one_hot(cls_codes.to(device), num_classes=16)

        # self.visualize(Path('debug'), voxel_grids, p, occ, batch.get('partial_aug'))
        input_features = self.model.infer_c(partial_input.transpose(1, 2))

        return self.model.compute_loss(input_features_for_completion=input_features, cls_codes_for_completion=cls_codes,
                                       input_points_for_completion=p, input_points_occ_for_completion=occ,
                                       voxel_grids=voxel_grids, balance_weight=self.balance_weight)

    def visualize(self, out_scan_dir, voxel_grids, p, occ, partial_pc):
        for i in range(voxel_grids.shape[0]):
            vox_to_mesh(voxel_grids[i].cpu().numpy(), out_scan_dir / f'{i}_voxel_grids', threshold=0.5)

            mesh_pts = p[i][occ[i] > 0]
            voxel_occ = pointcloud2voxel_fast(mesh_pts.unsqueeze(0))
            vox_to_mesh(voxel_occ[0].cpu().numpy(), out_scan_dir / f'{i}_voxel_occ', threshold=0.0)
            write_pointcloud(out_scan_dir / f'{i}_partial_pc.ply', partial_pc[i].cpu().numpy())

    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(epoch)
        if not os.path.exists(path):
            torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, path)

    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path + '/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0

        checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=int)
        checkpoints = np.sort(checkpoints)
        path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoints[-1])

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return epoch
