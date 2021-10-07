import argparse
import multiprocessing as mp
import os
import random
import traceback
from glob import glob
from multiprocessing import Pool

import numpy as np
import trimesh
from scipy.spatial import cKDTree as KDTree

from . import implicit_waterproofing as iw

ROOT = 'shapenet/data/'


class PointCloud2VoxelKDTree:
    def __init__(self, res=32, grid_size=1.):
        half_size = grid_size / 2.
        grid_points = iw.create_grid_points_from_bounds(-half_size, half_size, res)
        self.kdtree = KDTree(grid_points)
        self.grid_len = len(grid_points)
        self.res = res

    def __call__(self, point_cloud):
        occupancies = np.zeros(self.grid_len, dtype=np.int8)

        _, idx = self.kdtree.query(point_cloud)
        occupancies[idx] = 1
        occupancies.shape = (self.res,) * 3
        return occupancies


def voxelized_pointcloud_sampling(path):
    try:
        out_file = path + '/voxelized_point_cloud_{}res_{}points.npz'.format(args.res, args.num_points)

        if os.path.exists(out_file):
            print('File exists. Done.')
            return
        off_path = path + '/isosurf_scaled.off'

        mesh = trimesh.load(off_path)
        point_cloud = mesh.sample(args.num_points)

        occupancies = np.zeros(len(grid_points), dtype=np.int8)

        _, idx = kdtree.query(point_cloud)
        occupancies[idx] = 1

        compressed_occupancies = np.packbits(occupancies)

        np.savez(out_file, point_cloud=point_cloud, compressed_occupancies=compressed_occupancies, bb_min=bb_min,
                 bb_max=bb_max, res=args.res)
        print('Finished {}'.format(path))

    except Exception as err:
        print('Error with {}: {}'.format(path, traceback.format_exc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run point cloud sampling'
    )

    parser.add_argument('-res', type=int)
    parser.add_argument('-num_points', type=int)

    args = parser.parse_args()

    bb_min = -0.5
    bb_max = 0.5

    grid_points = iw.create_grid_points_from_bounds(bb_min, bb_max, args.res)
    kdtree = KDTree(grid_points)

    p = Pool(mp.cpu_count())
    paths = glob(ROOT + '/*/*/')

    # enabeling to run te script multiple times in parallel: shuffling the data
    random.shuffle(paths)
    p.map(voxelized_pointcloud_sampling, paths)
