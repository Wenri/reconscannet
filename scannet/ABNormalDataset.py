import sys
from collections import OrderedDict
from operator import itemgetter
from pathlib import Path

import numpy as np
import trimesh
from torch.utils import data

from if_net.models.data.core import list_categories
from scannet.scannet_utils import ShapeNetCat


class ABNormalDataset(data.Dataset):
    """ 3D Occupancy ABNormal dataset class.
    """

    def __init__(self, mode, cfg, valid_only=True):
        data_dir = Path(cfg['data']['abnormal'])
        scans = {}
        for sdir in ('red', 'black'):
            for it in data_dir.glob(f'*/{sdir}/scan_*/*_partial_pc.ply'):
                file_id = int(it.parent.name.split('_')[1]), int(it.name.split('_')[0])
                scans.setdefault(file_id, it)
        self.npz_files = OrderedDict(sorted(scans.items(), key=itemgetter(0)))
        scans = {}
        for it in data_dir.glob('*/gen/scan_*/*_output.npz'):
            file_id = int(it.parent.name.split('_')[1]), int(it.name.split('_')[0])
            scans.setdefault(file_id, it)
        self.anno_files = OrderedDict(sorted(scans.items(), key=itemgetter(0)))
        self.index_list = tuple(self.anno_files.keys() if valid_only else self.npz_files.keys())
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
        return len(self.index_list)

    def subsample(self, points, N):
        total = points.shape[0]
        indices = self.rand.permutation(total)
        if indices.shape[0] < N:
            indices = np.concatenate([indices, self.rand.integers(total, size=N - total)])
        indices = indices[:N]
        return points[indices]

    def load_partial(self, file_id):
        npz_name = self.npz_files[file_id]
        pc_red = trimesh.load(npz_name)
        # pc_black = trimesh.load(black_path)
        # assert np.allclose(pc_red.vertices, pc_black.vertices)
        return self.subsample(np.asarray(pc_red.vertices, dtype=np.float32), self.N)

    def get_cls(self, file):
        scan_dir = file.parent
        gen_dir = scan_dir.parent
        cls_dir = gen_dir.parent
        cls_name = cls_dir.name.split('_')[0]
        return self.catmap[cls_name]

    def load_by_id(self, scan_id, idx):
        file = self.npz_files.get((int(scan_id), int(idx)))
        for i, file in enumerate(self.npz_files):
            if f"scan_{scan_id}" != file.parent.name:
                continue
            if file.name.split('_')[0] != idx:
                continue
        return np.load(file), self.get_cls(file)

    def __getitem__(self, idx):
        while True:
            try:
                file_id = self.index_list[idx]
                anno_name = self.anno_files.get(file_id)
                n_in = self.OCCN // 2
                pts_mask = np.zeros(self.OCCN, dtype=np.bool_)
                pts_mask[:n_in] = True
                if anno_name:
                    npz_file = np.load(anno_name)
                    pts = npz_file['pts']
                    mask = npz_file['pts_mask']
                    inpts = pts[np.all(mask, axis=-1)]
                    outpts = pts[~mask[:, 0] & mask[:, 1]]

                    inpts = self.subsample(inpts, n_in)
                    outpts = self.subsample(outpts, self.OCCN - n_in)
                    indices = self.rand.permutation(self.OCCN)
                    pts = np.concatenate((inpts, outpts), axis=0)[indices]
                    pts_mask = pts_mask[indices]
                else:
                    anno_name = self.npz_files.get(file_id)
                    pts = np.zeros((self.OCCN, 3), dtype=np.float32)
                    n_in = 0

                ret = {
                    'pts': pts,
                    'pts_mask': pts_mask,
                    'partial_pc': self.load_partial(file_id),
                    'cls': self.get_cls(anno_name),
                    'n_in': n_in,
                    'idx': file_id
                }
                return ret
            except Exception as e:
                print('Error while loading data: ', e, file=sys.stderr)
                idx = (idx + 1) % len(self.npz_files)
