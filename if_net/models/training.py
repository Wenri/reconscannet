from __future__ import division

import os
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from export_scannet_pts import vox_to_mesh
from net_utils.voxel_util import pointcloud2voxel_fast


class Trainer(object):
    def __init__(self, model, device, exp_name, optimizer='Adam'):
        self.model = model.to(device)
        self.device = device
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        if optimizer == 'Adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters())
        if optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), momentum=0.9)

        self.exp_path = os.path.dirname(__file__) + '/../experiments/{}/'.format(exp_name)
        self.checkpoint_path = self.exp_path + 'checkpoints/'.format(exp_name)
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.writer = SummaryWriter(self.exp_path + 'summary'.format(exp_name))
        self.val_min = None

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss, _ = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_loss(self, batch):
        device = self.device

        partial = batch.get('partial').to(device)
        p = batch.get('points').to(device)
        occ = batch.get('points.occ').to(device)
        # voxel_gt = batch.get('voxels').to(device)
        voxel_grids = pointcloud2voxel_fast(partial)

        # self.visualize(Path('debug'), voxel_grids, p, occ)
        input_features = self.model.infer_c(partial.transpose(1, 2))

        return self.model.compute_loss(input_features_for_completion=input_features,
                                       input_points_for_completion=p,
                                       input_points_occ_for_completion=occ,
                                       voxel_grids=voxel_grids)

    def visualize(self, out_scan_dir, voxel_grids, p, occ):
        vox_to_mesh(voxel_grids[0].cpu().numpy(), out_scan_dir / 'voxel_grids', threshold=0.5)

        mesh_pts = p[0][occ[0] > 0]
        voxel_occ = pointcloud2voxel_fast(mesh_pts.unsqueeze(0))
        vox_to_mesh(voxel_occ[0].cpu().numpy(), out_scan_dir / 'voxel_occ', threshold=0.0)

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
