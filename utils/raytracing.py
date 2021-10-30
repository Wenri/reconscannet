import argparse
import os
import sys
import time
from collections import defaultdict
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


def get_labeled_face(mesh):
    face_colors = torch.from_numpy(mesh.visual.face_colors)
    face_mask = torch.all(torch.logical_and(
        face_colors >= torch.as_tensor((128, 0, 0, 0), dtype=face_colors.dtype),
        face_colors <= torch.as_tensor((255, 128, 128, 255), dtype=face_colors.dtype)), dim=-1)

    return face_mask


class PreCalcMesh:
    def __init__(self, m):
        assert m.is_watertight
        fix_normals(m)
        self.mesh = m
        self.face_mask = get_labeled_face(m)
        self.edges, self.edge_norms, selected_mask = self.extract_adj_faces()
        self.face_adjacency = torch.from_numpy(m.face_adjacency)[selected_mask]
        self.face_adjacency_edges = torch.from_numpy(m.face_adjacency_edges)[selected_mask]
        self.good_pts = self.extract_broken_pts()

    def extract_adj_faces(self):
        mesh = self.mesh

        selected_mask = torch.all(self.face_mask[mesh.face_adjacency], dim=-1)
        selected_mask = torch.logical_and(torch.from_numpy(mesh.face_adjacency_convex), selected_mask)

        vertices = torch.from_numpy(mesh.vertices)
        pairs = torch.from_numpy(mesh.face_normals)[torch.from_numpy(mesh.face_adjacency)[selected_mask]]
        edges = vertices[torch.from_numpy(mesh.face_adjacency_edges)[selected_mask]]
        unshared = vertices[torch.from_numpy(mesh.face_adjacency_unshared)[selected_mask]]

        edges_vec = edges[:, 0] - edges[:, 1]
        edges_vec /= torch.linalg.vector_norm(edges_vec, dim=-1, keepdim=True)
        new_norms = torch.cross(pairs, edges_vec.unsqueeze(1).expand(-1, 2, -1))

        test_sign = unshared - edges[:, 0].unsqueeze(1)
        test_sign = torch.sign(torch.matmul(test_sign.unsqueeze(2), new_norms.unsqueeze(3)).squeeze(3))
        new_norms *= test_sign

        return edges, new_norms, selected_mask

    def extract_broken_pts(self):
        broken_pts = frozenset(torch.from_numpy(self.mesh.faces)[torch.logical_not(self.face_mask)].flatten().tolist())
        good_pts = defaultdict(set)
        for a, b in self.mesh.edges_unique:
            if a not in broken_pts:
                good_pts[a.item()].add(b.item())
            if b not in broken_pts:
                good_pts[b.item()].add(a.item())

        return good_pts

    def check_is_verified(self, pts):
        v = self.face_mask
        mesh = self.mesh

        normals = -torch.from_numpy(mesh.face_normals)[v]
        triangles = torch.from_numpy(mesh.triangles)

        in_trig = TriangleRayIntersection(pts, normals, triangles[v, 0], triangles[v, 1], triangles[v, 2])
        # mesh.visual.face_colors[in_trig, :3] = np.array((0, 255, 0))
        return torch.any(in_trig, dim=-1)

    def check_is_edge(self, pts):
        edges = self.edges

        pts_vec = edges[:, 0] - pts

        proj_sign = pts_vec.unsqueeze(1).expand(-1, 2, -1)
        proj_sign = torch.matmul(proj_sign.unsqueeze(2), self.edge_norms.unsqueeze(3)).squeeze(3).squeeze(2)
        is_proj = torch.all(proj_sign > 0, dim=1)

        edges_vec = edges[is_proj, 0] - edges[is_proj, 1]
        edges_len = torch.linalg.vector_norm(edges_vec, dim=-1, keepdim=True)
        edges_vec /= edges_len
        proj_len = torch.matmul(pts_vec[is_proj].unsqueeze(1), edges_vec.unsqueeze(2)).squeeze(2).squeeze(1)
        is_length = torch.logical_and(proj_len >= 0, proj_len <= edges_len.squeeze(1))

        face_adjacency = self.face_adjacency[is_proj]
        face_adjacency_edges = self.face_adjacency_edges[is_proj]
        self.mesh.visual.face_colors[face_adjacency[is_length, 0].numpy(), :3] = np.array((0, 0, 255))
        self.mesh.visual.face_colors[face_adjacency[is_length, 1].numpy(), :3] = np.array((0, 255, 0))

        return torch.any(is_length).item() or self.check_is_extended(pts, face_adjacency_edges, proj_len)

    def check_is_extended(self, pts, edges, proj_len):
        if not len(edges):
            return False

        candidate_pts = torch.where(proj_len < 0, edges[:, 0], edges[:, 1])
        for idx in candidate_pts.tolist():
            tid = self.good_pts.get(idx)
            if tid is None:
                continue
            target_pts = torch.from_numpy(self.mesh.vertices[np.fromiter(tid, count=len(tid), dtype=np.int_)])
            center_pts = torch.from_numpy(self.mesh.vertices[idx])
            edges_vec = target_pts - center_pts
            edges_len = torch.linalg.vector_norm(edges_vec, dim=-1, keepdim=True)
            edges_vec /= edges_len
            pts_vec = pts - center_pts
            proj_len = torch.mm(edges_vec, pts_vec.unsqueeze(1)).squeeze(1)
            is_in = proj_len <= edges_len
            if torch.all(is_in):
                return True

        return False


def main(args):
    tol.facet_threshold = 5

    m = PreCalcMesh(trimesh.load(args.plyfile, force='mesh'))

    # full_labels = get_normal_cls(m.face_normals)
    lut = gen_lut()

    # for i, f in enumerate(m.facets):
    #     m.visual.face_colors[f, :3] = lut[(i + 1) % 256, 0]

    pts = torch.from_numpy(create_grid_points_from_bounds(-0.55, .55, 64))
    pts_split = torch.tensor_split(pts, 64)
    pts_mask = torch.cat([m.check_is_verified(p) for p in pts_split])
    pts_rev = torch.logical_not(pts_mask)
    spts = pts[pts_rev]
    pts_mask[pts_rev] = torch.from_numpy(np.fromiter(map(m.check_is_edge, spts), dtype=np.bool_, count=spts.shape[0]))

    print('Total Verified PTS:', torch.count_nonzero(pts_mask))

    write_pointcloud(args.plyfile.with_suffix('.pc.ply'), pts[np.logical_not(pts_mask)].numpy())
    m.mesh.export(args.plyfile.with_suffix('.cls.ply'))

    return os.EX_OK


if __name__ == '__main__':
    sys.exit(main(parse_args()))
