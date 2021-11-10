from __future__ import division

import os
from glob import glob
from pathlib import Path

import numpy as np
import torch
from pytorch3d.ops import knn_points
from torch.utils.tensorboard import SummaryWriter

from export_scannet_pts import vox_to_mesh, write_pointcloud
from external.common import compute_iou_cuda
from if_net.data_processing.voxelized_pointcloud_sampling import PointCloud2VoxelKDTree
from if_net.models.data.transforms import generate_rotmatrix
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
        self.voxgen = PointCloud2VoxelKDTree()
        _rot_matrix = torch.stack([torch.from_numpy(m.T.astype(np.float32)) for m in generate_rotmatrix()])
        self._rot_matrix_cuda = _rot_matrix.to(self.device)

    def verify_alignment(self, partial, full_pc, threshold=1e-2, max_error=5, return_valid=False):
        x_nn = knn_points(partial, full_pc, K=1, return_sorted=False).dists[..., 0]
        invalid_count = torch.count_nonzero(x_nn > threshold, dim=-1)
        if return_valid:
            selected_id = torch.nonzero(invalid_count <= max_error, as_tuple=True)[0]
        else:
            selected_id = torch.nonzero(invalid_count > max_error, as_tuple=True)[0]
        return selected_id, invalid_count[selected_id]

    def rotmatrix_match(self, partial, full_pc):
        n_batch = len(self._rot_matrix_cuda)
        partial = partial.unsqueeze(0).expand(n_batch, -1, -1)
        full_pc = full_pc.unsqueeze(0).expand(n_batch, -1, -1)

        partial = torch.bmm(partial, self._rot_matrix_cuda)
        valid_id, invalid_count = self.verify_alignment(partial, full_pc, return_valid=True)

        if not len(valid_id):
            return None

        idx = valid_id[torch.argmin(invalid_count)]
        return partial[idx]

    def try_fix_partial(self, partial, full_pc, rdist_threshold=0.1):
        invalid_id = set()
        fixed_id = set()
        for idx in self.verify_alignment(partial, full_pc)[0].tolist():
            partial_t = self.rotmatrix_match(partial[idx], full_pc[idx])
            if partial_t is not None:
                fixed_id.add(idx)
                partial[idx] = partial_t.squeeze(0)
            else:
                invalid_id.add(idx)

        x_nn = knn_points(full_pc, partial, K=1, return_sorted=False).dists[..., 0]
        dist = x_nn.mean(-1)
        invalid_id = {idx: dist[idx].item() for idx in invalid_id}
        for idx in torch.nonzero(dist > rdist_threshold, as_tuple=True)[0].tolist():
            invalid_id[idx] = dist[idx].item()
        return invalid_id, fixed_id

    def train_step(self, batch, **kwargs):
        self.model.train()
        self.optimizer.zero_grad()

        partial = batch.get('partial').to(self.device)
        full_pc = batch.get('pc').to(self.device)
        batch_size = partial.shape[0]

        invalid_id, fixed_id = self.try_fix_partial(partial, full_pc)
        valid_mask = torch.ones(size=(batch_size,), device=self.device)
        valid_mask[list(invalid_id.keys())] = 0

        loss, _ = self.compute_loss(
            partial=partial, valid_mask=valid_mask,
            p=batch.get('points').to(self.device),
            occ=batch.get('points.occ').to(self.device),
            cls_codes=batch.get('category'), **kwargs)

        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        partial_aug_count = torch.count_nonzero(batch.get('partial.aug')).item()
        return loss.item(), partial_aug_count, fixed_id, invalid_id

    def eval_step(self, batch):
        self.model.eval()

        partial = batch.get('partial').to(self.device)
        p = batch.get('points_iou').to(self.device)
        occ = batch.get('points_iou.occ').to(self.device)
        cls_codes = batch.get('category')
        full_pc = batch.get('pc').to(self.device)

        with torch.no_grad():
            invalid_id, fixed_id = self.try_fix_partial(partial, full_pc)
            c = self.model.infer_c(partial.transpose(1, 2), cls_codes_for_completion=cls_codes)
            z = self.model.get_z_from_prior((1,), sample=False, device=self.device)
            voxel_grids = pointcloud2voxel_fast(partial)
            occ_hat = self.model.decode(p, z, c, voxel_grids).logits
            iou_list = compute_iou_cuda(occ_hat, occ).tolist()

        return iou_list, invalid_id, fixed_id

    def overscan_aug(self, *partial, overscan_factor=0.02):
        device = self.device
        n_batch = partial[0].shape[0]

        overscan = torch.abs(torch.randn(n_batch, 1, 1, device=device)) * overscan_factor
        offset = (torch.rand(n_batch, 1, 3, device=device) - 0.5) * overscan

        return [(p + offset) / (overscan + 1) for p in partial]

    def compute_loss(self, partial, valid_mask, p, occ, cls_codes=None, **kwargs):
        partial_input, p = self.overscan_aug(partial, p)

        voxel_grids = pointcloud2voxel_fast(partial_input)
        # self.visualize('debug', voxel_grids, partial_input, p, occ)

        input_features = self.model.infer_c(partial_input.transpose(1, 2), cls_codes_for_completion=cls_codes)
        loss = self.model.compute_loss(input_features_for_completion=input_features, voxel_grids=voxel_grids,
                                       input_points_for_completion=p, input_points_occ_for_completion=occ,
                                       balance_weight=self.balance_weight, valid_mask=valid_mask)

        return loss

    def visualize(self, out_scan_dir, voxel_grids, partial_input, p, occ):
        out_scan_dir = Path(out_scan_dir)
        for i in range(voxel_grids.shape[0]):
            full_pc = p[i][occ[i] > 0].cpu().numpy()
            vox_to_mesh(voxel_grids[i].cpu().numpy(), out_scan_dir / f'{i}_voxel_grids', threshold=0.5)
            vox_to_mesh(self.voxgen(full_pc), out_scan_dir / f'{i}_voxel_pc', threshold=0.0)
            write_pointcloud(out_scan_dir / f'{i}_full_pc.ply', partial_input[i].cpu().numpy(),
                             rgb_points=np.asarray((0, 255, 0), dtype=np.uint8))

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

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            print(f'Resetting lr: {param_group["lr"]} initial_lr: {param_group.get("initial_lr")} to {lr}')
            param_group['lr'] = lr
