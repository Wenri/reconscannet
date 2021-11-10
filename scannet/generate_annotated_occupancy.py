import argparse
import os
import sys
from collections import namedtuple
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import trimesh
from termcolor import colored
from trimesh.repair import fix_normals

from export_scannet_pts import write_pointcloud
from if_net.data_processing.implicit_waterproofing import create_grid_points_from_bounds, implicit_waterproofing
from scannet.mesh_csv_register import MeshRegister
from utils.raytracing import PreCalcMesh, NearestMeshQuery

COLOR_BOUND = namedtuple('COLOR_BOUND', ('color_low', 'color_high'))
LabeledColor = namedtuple('LABELED_COLOR', ('Red', 'Black'))(
    Red=COLOR_BOUND((180, 0, 0, 0), (255, 50, 50, 255)),
    Black=COLOR_BOUND((0, 0, 0, 0), (50, 50, 50, 255))
)


class AnnotatedMesh:
    def __init__(self, args, mesh_file, scan_info, device, **kwargs):
        m_red = self.load_mesh(args.red_path / mesh_file)
        m_black = self.load_mesh(args.black_path / mesh_file)
        if m_red is None or m_black is None:
            raise RuntimeError('load failed, maybe not watertight!')
        assert np.allclose(m_red.vertices, m_black.vertices)
        assert np.allclose(m_red.faces, m_black.faces)
        self.device = device
        red_mesh = PreCalcMesh(m_red, device, **LabeledColor.Red._asdict(), **kwargs)
        black_mesh = NearestMeshQuery(m_black, device, **LabeledColor.Black._asdict())
        self.mesh = red_mesh.mesh
        self.red_mesh = red_mesh if red_mesh else None
        self.black_mesh = black_mesh if black_mesh else None
        self.scan_info = scan_info

        if not m_red.is_watertight or not m_black.is_watertight:
            print('NW!', end='!')
        if scan_info.PerfectA:
            print('PerfectA', end=' ' if black_mesh else '+')
        if scan_info.PerfectB and red_mesh:
            print('PerfectB', end=' ' if red_mesh else '+')
        if scan_info.Trusted:
            print('Trusted', end=' ' if red_mesh else '!')

    def load_mesh(self, file_path):
        ml = [m for m in trimesh.load(file_path, force='mesh', process=False).split(only_watertight=False)
              if len(m.faces) > 10]
        ml = trimesh.util.concatenate(ml)
        if ml.fill_holes():
            fix_normals(ml)
        return ml

    def query_pts(self, pts):
        contains, holes_list = implicit_waterproofing(self.mesh, pts.cpu().numpy())
        if np.any(holes_list):
            print('holes_list: ', holes_list.nonzero())
        contains_pos = torch.from_numpy(contains).to(device=self.device)
        contains_ok = torch.from_numpy(contains).to(device=self.device)
        contains_neg = torch.logical_not(contains_pos)
        to_red_pts = pts[contains_neg]
        to_black_pts = pts[contains_pos]
        if len(to_red_pts):
            if self.scan_info.PerfectB or (self.red_mesh is None and self.scan_info.Trusted):
                to_red_mask = torch.ones(to_red_pts.shape[0], dtype=torch.bool, device=self.device)
            elif self.red_mesh is not None:
                to_red_mask = self.red_mesh.check_is_verified(to_red_pts)
                to_red_rev = torch.logical_not(to_red_mask)
                to_red_mask[to_red_rev] = self.red_mesh.check_is_edge(to_red_pts[to_red_rev])
            else:
                raise RuntimeError('Untrusted mesh without Red Label!')
            contains_ok[contains_neg] = to_red_mask
        if not self.scan_info.PerfectA and len(to_black_pts):
            if self.black_mesh is not None:
                to_black_mask = torch.from_numpy(self.black_mesh.check_is_black(to_black_pts)).to(device=self.device)
            else:
                raise RuntimeError('Imperfect mesh without Black Label!')
            contains_pos[contains] ^= to_black_mask

        return torch.stack((contains_pos, contains_ok), dim=-1)


def main(args):
    device = torch.device('cuda')
    pool = Pool()

    args.red_path = args.data_dir / 'red'
    args.black_path = args.data_dir / 'black'
    gen_path = args.data_dir / 'gen'

    register = MeshRegister(args)

    black_files = set(os.fspath(a.relative_to(args.black_path)) for a in args.black_path.glob('scan_*/*_output.ply'))
    red_files = set(os.fspath(a.relative_to(args.red_path)) for a in args.red_path.glob('scan_*/*_output.ply'))
    all_files = list(red_files.intersection(black_files))
    all_files.sort(key=lambda x: (int(os.path.dirname(x).split('_')[-1]), int(os.path.basename(x).split('_')[0])))

    pts = torch.from_numpy(create_grid_points_from_bounds(-0.55, .55, 64)).to(device=device, dtype=torch.float)
    pts_split = torch.tensor_split(pts, 64)

    err = 0
    consistency = 0
    unknown = 0
    for file in all_files:
        try:
            scan_info = register.check_scan(os.path.dirname(file), os.path.basename(file))
            if not scan_info.Usable:
                print(file, ' Unusable Skip')
                continue
            print(file, end=' ')
            m = AnnotatedMesh(args, file, scan_info, device, pool=pool)
            pts_mask = torch.cat([m.query_pts(p) for p in pts_split])
            inpts = pts[torch.all(pts_mask, dim=-1)]
            outpts = pts[~pts_mask[:, 0] & pts_mask[:, 1]]
            mesh_file = gen_path / file
            mesh_file.parent.mkdir(exist_ok=True)
            write_pointcloud(mesh_file.with_suffix('.in.ply'), inpts.cpu().numpy())
            write_pointcloud(mesh_file.with_suffix('.out.ply'), outpts.cpu().numpy(),
                             rgb_points=np.asarray((0, 0, 255), dtype=np.uint8))
            np.savez(mesh_file.with_suffix('.npz'), pts=pts.cpu().numpy(),
                     pts_mask=pts_mask.cpu().numpy())
            print('OK')
        except RuntimeError as e:
            print(colored('RuntimeError: ', 'magenta'), e)
            err += 1
        except AssertionError as e:
            print(colored('AssertionError: ', 'magenta'), e)
            consistency += 1
        except Exception as e:
            print(colored('UnknownError: ', 'magenta'), e)
            unknown += 1

    print(f'{err=}, {consistency=}, {unknown=}, total={len(all_files)}')
    return os.EX_OK


def parse_args():
    """PARAMETERS"""
    base_dir = Path(__file__).parent
    parser = argparse.ArgumentParser('Instance Scene Completion.')
    parser.add_argument('--max_samples', type=int, default=60000, help='number of points')
    parser.add_argument('--data_dir', type=Path, help='optional reload model path', default=Path(
        'data', 'anno1', 'chair'))
    parser.add_argument('--csv_file', type=Path, help='optional reload model path', default=Path(
        'data', 'anno1', 'chair', 'chair.csv'))
    return parser.parse_args()


if __name__ == '__main__':
    sys.exit(main(parse_args()))
