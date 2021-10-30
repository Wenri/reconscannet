import argparse
import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import trimesh
from trimesh.constants import tol
from trimesh.repair import fix_normals

import check_is_extended
from TriangleRayIntersection import TriangleRayIntersection
from export_scannet_pts import write_pointcloud
from if_net.data_processing.implicit_waterproofing import create_grid_points_from_bounds


def parse_args():
    """PARAMETERS"""
    base_dir = Path(__file__).parent
    parser = argparse.ArgumentParser('Instance Scene Completion.')
    parser.add_argument('--max_samples', type=int, default=60000, help='number of points')
    parser.add_argument('--plyfile', type=Path, help='optional reload model path', default=Path(
        'data', 'label_1018', 'scan_64', '9_1be38f2624022098f71e06115e9c3b3e_output.ply'))
    return parser.parse_args()


def get_labeled_face(mesh, device):
    face_colors = torch.from_numpy(mesh.visual.face_colors).to(device, dtype=torch.uint8)
    face_mask = torch.all(torch.logical_and(
        face_colors >= torch.as_tensor((128, 0, 0, 0), dtype=face_colors.dtype, device=device),
        face_colors <= torch.as_tensor((255, 128, 128, 255), dtype=face_colors.dtype, device=device)), dim=-1)

    return face_mask


class PreCalcMesh:
    def __init__(self, m, device):
        assert m.is_watertight
        fix_normals(m)
        self.device = device
        self.mesh = m
        self.face_mask = get_labeled_face(m, device)
        self.edges, self.edge_norms, selected_mask = self.extract_adj_faces()
        self.face_adjacency = torch.from_numpy(m.face_adjacency)[selected_mask]
        self.face_adjacency_edges = torch.from_numpy(m.face_adjacency_edges)[selected_mask]

        check_is_extended.vertices = np.asarray(m.vertices)
        check_is_extended.good_pts = self.extract_broken_pts()

        self.pool = Pool()

    def extract_adj_faces(self):
        mesh = self.mesh
        device = self.device

        selected_mask = torch.all(self.face_mask[mesh.face_adjacency], dim=-1)
        selected_mask = torch.logical_and(torch.from_numpy(mesh.face_adjacency_convex).to(device=device), selected_mask)

        vertices = torch.from_numpy(mesh.vertices).to(device=device, dtype=torch.float)
        edges = vertices[torch.from_numpy(mesh.face_adjacency_edges).to(device=device)[selected_mask]]
        unshared = vertices[torch.from_numpy(mesh.face_adjacency_unshared).to(device=device)[selected_mask]]
        face_adjacency = torch.from_numpy(mesh.face_adjacency).to(device=device)[selected_mask]
        pairs = torch.from_numpy(mesh.face_normals).to(device=device, dtype=torch.float)[face_adjacency]

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

        normals = torch.neg(torch.from_numpy(mesh.face_normals).to(device=self.device, dtype=torch.float)[v])
        triangles = torch.from_numpy(mesh.triangles).to(device=self.device, dtype=torch.float)

        in_trig = TriangleRayIntersection(pts, normals, triangles[v, 0], triangles[v, 1], triangles[v, 2])
        # mesh.visual.face_colors[in_trig, :3] = np.array((0, 255, 0))
        return torch.any(in_trig, dim=-1)

    def check_is_edge(self, pts):
        edges = self.edges

        pts_vec = edges[:, 0].unsqueeze(0) - pts.unsqueeze(1)

        proj_sign = torch.sum(torch.multiply(pts_vec.unsqueeze(2), self.edge_norms), dim=-1)
        is_proj = torch.all(proj_sign > 0, dim=-1)

        edges_vec = edges[:, 0] - edges[:, 1]
        edges_len = torch.linalg.vector_norm(edges_vec, dim=-1, keepdim=True)
        edges_vec /= edges_len
        proj_len = torch.sum(torch.multiply(pts_vec, edges_vec), dim=-1)
        is_length = torch.logical_and(proj_len >= 0, proj_len <= edges_len.squeeze(-1))

        is_ok = torch.logical_and(is_proj, is_length)

        # active_face = torch.any(is_ok, dim=0)
        # self.mesh.visual.face_colors[self.face_adjacency[active_face, 0].numpy(), :3] = np.array((0, 0, 255))
        # self.mesh.visual.face_colors[self.face_adjacency[active_face, 1].numpy(), :3] = np.array((0, 255, 0))

        pts_mask = torch.any(is_ok, dim=-1)

        pts_list = torch.logical_not(pts_mask).nonzero(as_tuple=True)[0].tolist()

        params = ((pts[i].cpu().numpy(),
                   self.face_adjacency_edges[is_proj[i]].numpy(),
                   proj_len[i, is_proj[i]].cpu().numpy()) for i in pts_list)

        map_ret = self.pool.starmap(check_is_extended.check_is_extended, params)
        pts_mask[pts_list] = torch.as_tensor(map_ret, dtype=pts_mask.dtype, device=pts_mask.device)

        return pts_mask


def main(args):
    tol.facet_threshold = 5
    device = torch.device('cuda')

    m = PreCalcMesh(trimesh.load(args.plyfile, force='mesh'), device=device)

    pts = torch.from_numpy(create_grid_points_from_bounds(-0.55, .55, 64)).to(device=device, dtype=torch.float)
    pts_split = torch.tensor_split(pts, 2)
    pts_mask = torch.cat([m.check_is_verified(p) for p in pts_split])
    pts_rev = torch.logical_not(pts_mask)
    pts_split = torch.tensor_split(pts[pts_rev], 2)
    pts_mask[pts_rev] = torch.cat([m.check_is_edge(p) for p in pts_split])

    print('Total Verified PTS:', torch.count_nonzero(pts_mask).item())

    write_pointcloud(args.plyfile.with_suffix('.pc.ply'), pts[torch.logical_not(pts_mask)].cpu().numpy())
    m.mesh.export(args.plyfile.with_suffix('.cls.ply'))

    return os.EX_OK


if __name__ == '__main__':
    sys.exit(main(parse_args()))
