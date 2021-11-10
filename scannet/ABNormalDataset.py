from pathlib import Path

import numpy as np
import trimesh
from torch.utils import data


class ABNormalDataset(data.Dataset):
    """ 3D Occupancy ABNormal dataset class.
    """

    def __init__(self, mode, cfg):
        data_dir = Path(cfg['data']['abnormal'])
        self.npz_files = list(data_dir.glob('*/gen/scan_*/*_output.npz'))
        self.npz_files.sort()
        self.rand = np.random.default_rng()
        self.N = cfg['data']['pointcloud_n']
        self.OCCN = cfg['data']['points_subsample']

    def __len__(self):
        """ Returns the length of the dataset.
        """
        return len(self.npz_files)

    def subsample(self, points, N):
        total = points.shape[0]
        indices = self.rand.permutation(total)
        if indices.shape[0] < N:
            indices = np.concatenate([indices, self.rand.integers(total, size=N - total)])
        indices = indices[:N]
        return points[indices]

    def load_partial(self, idx):
        npz_name = self.npz_files[idx]
        file_name = npz_name.name
        scan_dir = npz_name.parent
        gen_dir = scan_dir.parent
        red_path = Path(gen_dir.parent, 'red', scan_dir.name, file_name[:-10] + 'partial_pc.ply')
        black_path = Path(gen_dir.parent, 'black', scan_dir.name, file_name[:-10] + 'partial_pc.ply')
        pc_red = trimesh.load(red_path if red_path.exists() else black_path)
        # pc_black = trimesh.load(black_path)
        # assert np.allclose(pc_red.vertices, pc_black.vertices)
        return self.subsample(np.asarray(pc_red.vertices, dtype=np.float32), self.N)

    def __getitem__(self, idx):
        npz_file = np.load(self.npz_files[idx])
        pts = npz_file['pts']
        pts_mask = npz_file['pts_mask']
        inpts = pts[np.all(pts_mask, axis=-1)]
        outpts = pts[~pts_mask[:, 0] & pts_mask[:, 1]]

        n_in = self.OCCN // 2
        inpts = self.subsample(inpts, n_in)
        outpts = self.subsample(outpts, self.OCCN - n_in)
        indices = self.rand.permutation(self.OCCN)
        pts = np.concatenate((inpts, outpts), axis=0)
        pts_mask = np.zeros(self.OCCN, dtype=np.bool_)
        pts_mask[:n_in] = True

        ret = {
            'pts': pts[indices],
            'pts_mask': pts_mask[indices],
            'partial_pc': self.load_partial(idx)
        }
        return ret
