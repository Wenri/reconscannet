import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
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


def extract_adj_faces(mesh, selected_mask):
    selected_mask = np.any(selected_mask[mesh.face_adjacency], axis=-1)
    selected_mask = np.logical_and(mesh.face_adjacency_convex, selected_mask)

    vertices = torch.from_numpy(mesh.vertices)
    pairs = torch.from_numpy(mesh.face_normals)[mesh.face_adjacency[selected_mask]]
    edges = vertices[mesh.face_adjacency_edges[selected_mask]]
    unshared = vertices[mesh.face_adjacency_unshared[selected_mask]]
    edges_vec = edges[:, 0] - edges[:, 1]
    edges_vec /= torch.linalg.vector_norm(edges_vec, dim=-1, keepdim=True)
    new_norms = torch.cross(pairs, edges_vec.unsqueeze(1).expand(-1, 2, -1))

    test_sign = unshared - edges[:, 0].unsqueeze(1)
    test_sign = torch.sign(torch.matmul(test_sign.unsqueeze(2), new_norms.unsqueeze(3)).squeeze(3))
    new_norms *= test_sign

    return edges, new_norms, mesh.face_adjacency[selected_mask]


def extract_broken_pts(mesh, selected_mask):
    broken_pts = frozenset(mesh.faces[np.logical_not(selected_mask)].flat)
    return broken_pts


def check_is_verified(pts, mesh, v):
    dir = -mesh.face_normals[v]
    pts = np.broadcast_to(pts, shape=dir.shape)

    in_trig = TriangleRayIntersection(pts, dir, mesh.triangles[v, 0], mesh.triangles[v, 1], mesh.triangles[v, 2])
    # mesh.visual.face_colors[in_trig, :3] = np.array((0, 255, 0))
    return np.any(in_trig)


def check_is_edge(pts, edges, edge_norms, broken_pts, face_adjacency, mesh):
    pts_vec = edges[:, 0] - pts

    edges_vec = edges[:, 0] - edges[:, 1]
    edges_len = torch.linalg.vector_norm(edges_vec, dim=-1, keepdim=True)
    edges_vec /= edges_len

    proj_len = torch.matmul(pts_vec.unsqueeze(1), edges_vec.unsqueeze(2)).squeeze(2).squeeze(1)
    is_length = torch.logical_and(proj_len >= 0, proj_len <= edges_len.squeeze(1))

    proj_sign = pts_vec.unsqueeze(1).expand(-1, 2, -1)
    proj_sign = torch.matmul(proj_sign.unsqueeze(2), edge_norms.unsqueeze(3)).squeeze(3).squeeze(2)
    is_proj = torch.all(proj_sign > 0, dim=1)

    in_edge = torch.logical_and(is_length, is_proj)
    mesh.visual.face_colors[face_adjacency[in_edge, 0], :3] = np.array((0, 0, 255))
    mesh.visual.face_colors[face_adjacency[in_edge, 1], :3] = np.array((0, 255, 0))

    return torch.any(in_edge).item()


def get_labeled_face(mesh):
    face_color = np.all(np.logical_and(
        mesh.visual.face_colors > np.array((128, -1, -1, -1), dtype=np.int_),
        mesh.visual.face_colors <= np.array((255, 128, 128, 255), dtype=np.int_)), axis=-1)

    return face_color


def main(args):
    tol.facet_threshold = 5

    m = trimesh.load(args.plyfile, force='mesh')
    assert m.is_watertight

    fix_normals(m)
    face_mask = get_labeled_face(m)
    edges, edge_norms, adj = extract_adj_faces(m, face_mask)
    broken_pts = extract_broken_pts(m, face_mask)

    # full_labels = get_normal_cls(m.face_normals)
    lut = gen_lut()

    # for i, f in enumerate(m.facets):
    #     m.visual.face_colors[f, :3] = lut[(i + 1) % 256, 0]

    pts = create_grid_points_from_bounds(-0.55, .55, 32)
    pts_list = np.fromiter((check_is_verified(p, m, face_mask) for p in pts), dtype=np.bool_, count=pts.shape[0])
    edge_list = np.fromiter((check_is_edge(p, edges, edge_norms, broken_pts, adj, m) for p in pts),
                            dtype=np.bool_, count=pts.shape[0])

    pts_mask = np.logical_or(edge_list, pts_list)
    print('Total Verified PTS:', np.count_nonzero(pts_mask))

    write_pointcloud(args.plyfile.with_suffix('.pc.ply'), pts[np.logical_not(pts_mask)])
    m.export(args.plyfile.with_suffix('.cls.ply'))

    return os.EX_OK


if __name__ == '__main__':
    sys.exit(main(parse_args()))
