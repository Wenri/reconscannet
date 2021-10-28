import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from sklearn.cluster import k_means
from sklearn.manifold import spectral_embedding
from sklearn.metrics.pairwise import cosine_similarity
from trimesh.constants import tol
from trimesh.repair import fix_normals

from export_scannet_pts import write_pointcloud
from if_net.data_processing.implicit_waterproofing import create_grid_points_from_bounds
from utils.TriangleRayIntersection import TriangleRayIntersection

CLUSTER_DIM = 3


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name, )
        print('Elapsed: %s' % (time.time() - self.tstart))


def gen_lut():
    """
    Generate a label colormap compatible with opencv lookup table, based on
    Rick Szelski algorithm in `Computer Vision: Algorithms and Applications`,
    appendix C2 `Pseudocolor Generation`.
    :Returns:
      color_lut : opencv compatible color lookup table
    """
    tobits = lambda x, o: np.array(list(np.binary_repr(x, 24)[o::-3]), np.uint8)
    arr = np.arange(256)
    r = np.concatenate([np.packbits(tobits(x, -3)) for x in arr])
    g = np.concatenate([np.packbits(tobits(x, -2)) for x in arr])
    b = np.concatenate([np.packbits(tobits(x, -1)) for x in arr])
    return np.concatenate([[[b]], [[g]], [[r]]]).T


def get_normal_cls(a):
    n_samples, _ = a.shape

    with Timer("abs_cosine_similarity"):
        sim_mat = np.abs(cosine_similarity(a))
    with Timer("spectral_embedding"):
        embedding = spectral_embedding(sim_mat, n_components=CLUSTER_DIM, drop_first=False)
    with Timer("k_means"):
        centers, labels, _ = k_means(embedding, n_clusters=CLUSTER_DIM)

    ret = [np.count_nonzero(labels == i) for i in range(CLUSTER_DIM)]
    max_lbl_id = np.argmax(ret)
    ret.sort(reverse=True)
    print('\t'.join(str(s) for s in ret))

    print(centers)
    plt.figure()
    plt.scatter(embedding[:, 1], embedding[:, 2], c=labels)
    plt.show()

    # dist = euclidean_distances(embedding, centers)
    # min_norm_ids = np.argmin(dist, axis=0)
    # best_norm_id = min_norm_ids[max_lbl_id]

    return labels


def parse_args():
    """PARAMETERS"""
    base_dir = Path(__file__).parent
    parser = argparse.ArgumentParser('Instance Scene Completion.')
    parser.add_argument('--max_samples', type=int, default=60000, help='number of points')
    parser.add_argument('--plyfile', type=Path, help='optional reload model path', default=Path(
        'data', 'label_1018', 'scan_64', '9_1be38f2624022098f71e06115e9c3b3e_output.ply'))
    return parser.parse_args()


def check_is_verified(pts, mesh):
    dir = -mesh.face_normals
    pts = np.broadcast_to(pts, shape=dir.shape)

    return TriangleRayIntersection(pts, dir, mesh.triangles[:, 0], mesh.triangles[:, 1], mesh.triangles[:, 2])


def main(args):
    tol.facet_threshold = 5

    m = trimesh.load(args.plyfile, force='mesh')
    assert m.is_watertight

    fix_normals(m)

    # full_labels = get_normal_cls(m.face_normals)
    lut = gen_lut()

    for i, f in enumerate(m.facets):
        m.visual.face_colors[f, :3] = lut[(i + 1) % 256, 0]

    m.export(args.plyfile.with_suffix('.cls.ply'))

    pts = create_grid_points_from_bounds(-0.55, 0.55, 32)
    pts_list = np.fromiter((np.any(check_is_verified(p, m)) for p in pts), dtype=np.bool_, count=pts.shape[0])

    print('Total Verified PTS:', np.count_nonzero(pts_list))

    write_pointcloud(args.plyfile.with_suffix('.pc.ply'), pts[np.logical_not(pts_list)])

    return os.EX_OK


if __name__ == '__main__':
    sys.exit(main(parse_args()))
