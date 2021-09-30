import os
from glob import glob

import mcubes
import numpy as np
import torch
import trimesh

from ..data_processing import implicit_waterproofing as iw


class Generator(object):
    def __init__(self, model, threshold, exp_name, checkpoint=None, device=torch.device("cuda"), resolution=16,
                 batch_points=1000000):
        self.model = model.to(device)
        self.model.eval()
        self.threshold = threshold
        self.device = device
        self.resolution = resolution
        self.checkpoint_path = 'trained_model'
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)
        self.batch_points = batch_points

        self.min = -0.5
        self.max = 0.5

        grid_points = iw.create_grid_points_from_bounds(self.min, self.max, self.resolution)
        grid_points[:, 0], grid_points[:, 2] = grid_points[:, 2], grid_points[:, 0].copy()

        a = self.max + self.min
        b = self.max - self.min

        grid_coords = grid_points - a
        grid_coords = grid_coords / b

        grid_coords = torch.from_numpy(grid_coords).to(self.device, dtype=torch.float)
        grid_coords = torch.reshape(grid_coords, (1, len(grid_points), 3)).to(self.device)
        self.grid_points_split = torch.split(grid_coords, self.batch_points, dim=1)

    def generate_mesh(self, *inputs):

        logits_list = []
        for points in self.grid_points_split:
            with torch.no_grad():
                logits = self.model(points, *inputs)
            logits_list.append(logits.squeeze(0))

        logits = torch.cat(logits_list, dim=0)

        return logits

    def mesh_from_logits(self, logits):
        logits = np.reshape(logits, (self.resolution,) * 3)

        # padding to ba able to retrieve object close to bounding box bondary
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        logits = np.pad(logits, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=threshold)

        vertices, triangles = mcubes.marching_cubes(
            logits, threshold)

        # remove translation due to padding
        vertices -= 1

        # rescale to original scale
        step = (self.max - self.min) / (self.resolution - 1)
        vertices = np.multiply(vertices, step)
        vertices += [self.min, self.min, self.min]

        return trimesh.Trimesh(vertices, triangles)

    def load_checkpoint(self, checkpoint):
        if checkpoint is None:
            checkpoints = glob(self.checkpoint_path + '/*')
            if len(checkpoints) == 0:
                print('No checkpoints found at {}'.format(self.checkpoint_path))

            checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
            checkpoints = np.array(checkpoints, dtype=int)
            checkpoints = np.sort(checkpoints)
            path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoints[-1])
        else:
            path = checkpoint
        print('Loaded checkpoint from: {}'.format(path))
        state_dict = {k[len('module.'):]: v for k, v in torch.load(path)['model_state_dict'].items()}
        self.model.load_state_dict(state_dict)
