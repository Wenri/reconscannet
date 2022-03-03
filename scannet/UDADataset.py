import sys
from operator import itemgetter
from pathlib import Path

import numpy as np
import trimesh
from torch.utils import data

from if_net.models.data.core import list_categories
from scannet.scannet_utils import ShapeNetCat


class UDADataset(data.Dataset):
    """ 3D Occupancy ABNormal dataset class.
    """

    @staticmethod
    def scan_files(data_dir):
        scans = {}

        for it in data_dir.glob(f'*/scan_*/*_partial_pc.ply'):
            file_id = int(it.parent.name.split('_')[1]), int(it.name.split('_')[0])
            scans.setdefault(file_id, it)

        files = sorted(scans.items(), key=itemgetter(0))
        return [a[1] for a in files]

    def __init__(self, mode, cfg):
        self.ply_files = self.scan_files(Path(cfg['data']['abnormal']))
        self.rand = np.random.default_rng()
        self.N = cfg['data']['pointcloud_n']
        self.OCCN = cfg['data']['points_subsample']
        categories = {c: idx for idx, c in enumerate(list_categories(cfg['data']['path']))}
        self.catmap = {}
        for key, catids in vars(ShapeNetCat).items():
            if key[-4:] != '_cat':
                continue
            c = None
            for i in catids:
                c0 = categories.get(i)
                if c0 is not None:
                    assert c is None
                    c = c0
            self.catmap[key[:-4]] = c

    def __len__(self):
        """ Returns the length of the dataset.
        """
        return len(self.ply_files)

    def subsample(self, points, N):
        total = points.shape[0]
        indices = self.rand.permutation(total)
        if indices.shape[0] < N:
            indices = np.concatenate([indices, self.rand.integers(total, size=N - total)])
        indices = indices[:N]
        return points[indices]

    def load_partial(self, idx):
        pc_red = trimesh.load(self.ply_files[idx])
        # pc_black = trimesh.load(black_path)
        # assert np.allclose(pc_red.vertices, pc_black.vertices)
        return self.subsample(np.asarray(pc_red.vertices, dtype=np.float32), self.N)

    def get_cls(self, idx):
        npz_name = self.ply_files[idx]
        scan_dir = npz_name.parent
        cls_dir = scan_dir.parent
        cls_name = cls_dir.name.split('_')[0]
        return self.catmap[cls_name]

    def get_id(self, idx):
        npz_name: Path = self.ply_files[idx]
        scan_dir = npz_name.parent
        ins_id, *_ = npz_name.name.split('_', maxsplit=1)
        _, scan_id = scan_dir.name.split('_', maxsplit=1)
        id_tuple = int(scan_id), int(ins_id)
        return id_tuple

    def __getitem__(self, idx):
        while True:
            try:
                ret = {
                    'partial_pc': self.load_partial(idx),
                    'cls': self.get_cls(idx),
                    'idx': self.get_id(idx)
                }
                return ret
            except Exception as e:
                print('Error while loading data: ', e, file=sys.stderr)
                idx = (idx + 1) % len(self)
