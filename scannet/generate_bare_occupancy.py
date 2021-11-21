import argparse
import os
import sys
from collections import namedtuple
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import plyfile
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
    Red=COLOR_BOUND((150, 0, 0, 0), (255, 150, 150, 255)),
    Black=COLOR_BOUND((0, 0, 0, 0), (100, 100, 100, 255))
)


class AnnotatedMesh:
    def __init__(self, args, mesh_file, scan_info, device, **kwargs):
        m_red, m_red_n, m_red_c = self.load_mesh(args.red_path / mesh_file)
        m_black, m_black_n, m_black_c = self.load_mesh(args.black_path / mesh_file)
        if m_red is None or m_black is None:
            raise RuntimeError('load failed, maybe not watertight!')
        mis_match = np.count_nonzero(~np.all(np.isclose(m_red.vertices, m_black.vertices, atol=1e-5), axis=-1))
        if mis_match:
            print('Mis_match:', mis_match, end=' ')
            if mis_match > 10:
                raise RuntimeError('Mesh Vertices Index mismatch!')
        assert np.allclose(m_red.faces, m_black.faces), 'Mesh Faces Index mismatch'
        self.device = device
        self.mesh = m_red
        if not m_red_n and m_black_n:
            print('B-N>R', end=' ')
            self.mesh = m_black
            m_red_visual = m_red.visual.copy()
            m_red = m_black.copy()
            m_red_visual.mesh = m_red
            m_red.visual = m_red_visual

        red_mesh = PreCalcMesh(m_red, device, **LabeledColor.Red._asdict(), **kwargs)
        black_mesh = NearestMeshQuery(m_black, device, **LabeledColor.Black._asdict())
        self.red_mesh = red_mesh if red_mesh else None
        self.black_mesh = black_mesh if black_mesh else None
        self.scan_info = scan_info

        if not m_red.is_watertight or not m_black.is_watertight:
            print('NW!', end='!')
        if scan_info.PerfectA:
            print('PerfectA', end=' ' if black_mesh else '+')
        if scan_info.PerfectB:
            print('PerfectB', end=' ' if red_mesh else '+')
        if scan_info.Trusted:
            print('Trusted', end=' ' if red_mesh else '!')

        if not black_mesh and not scan_info.PerfectA:
            raise RuntimeError('Imperfect mesh without Black Label!')
        if not red_mesh and not scan_info.Trusted:
            raise RuntimeError('Untrusted mesh without Red Label!')

    def load_mesh(self, file_path):
        ply = plyfile.PlyData.read(file_path)
        comments = ply.comments
        properties = set(p.name for p in ply['vertex'].properties)
        has_normal = 'nx' in properties and 'ny' in properties and 'nz' in properties
        ml = [m for m in trimesh.load(file_path, force='mesh', process=False).split(only_watertight=False)
              if len(m.faces) > 10]
        ml = trimesh.util.concatenate(ml)
        if ml.fill_holes():
            fix_normals(ml)
        return ml, has_normal, comments

    def query_pts(self, pts):
        contains, holes_list = implicit_waterproofing(self.mesh, pts.cpu().numpy())
        if np.any(holes_list):
            print('holes_list: ', holes_list.nonzero())
        contains_pos = torch.from_numpy(contains).to(device=self.device)
        contains_ok = torch.ones_like(contains_pos)
        contains_neg = torch.logical_not(contains_pos)
        if torch.any(contains_neg):
            if self.scan_info.PerfectB or (self.red_mesh is None and self.scan_info.Trusted):
                pass
            elif self.red_mesh is not None:
                pass
            else:
                raise RuntimeError('Untrusted mesh without Red Label!')
        if not self.scan_info.PerfectA and torch.any(contains_pos):
            if self.black_mesh is not None:
                pass
            else:
                raise RuntimeError('Imperfect mesh without Black Label!')

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
                print(file, '-->Unusable Skip')
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
    parser.add_argument('--start_from', type=int, help='starting from scan id', default=0)
    return parser.parse_args()


if __name__ == '__main__':
    sys.exit(main(parse_args()))
