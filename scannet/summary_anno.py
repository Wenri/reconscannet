import argparse
import os
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import plyfile
import torch
import trimesh
from termcolor import colored
from trimesh.repair import fix_normals

from scannet.mesh_csv_register import MeshRegister
from utils.raytracing import PreCalcMesh, NearestMeshQuery

COLOR_BOUND = namedtuple('COLOR_BOUND', ('color_low', 'color_high'))
LabeledColor = namedtuple('LABELED_COLOR', ('Red', 'Black'))(
    Red=COLOR_BOUND((150, 0, 0, 0), (255, 150, 150, 255)),
    Black=COLOR_BOUND((0, 0, 0, 0), (100, 100, 100, 255))
)


class AnnotatedMeshSummary:
    def __init__(self, args, mesh_file, scan_info):
        m_red, m_red_n, m_red_c = self.load_mesh(args.red_path / mesh_file)
        m_black, m_black_n, m_black_c = self.load_mesh(args.black_path / mesh_file)
        device = torch.device('cpu')
        if m_red is None or m_black is None:
            raise RuntimeError('load failed, maybe not watertight!')
        mis_match = np.count_nonzero(~np.all(np.isclose(m_red.vertices, m_black.vertices, atol=1e-5), axis=-1))
        if mis_match:
            print('Mis_match:', mis_match, end=' ')
            if mis_match > 10:
                raise RuntimeError('Mesh Vertices Index mismatch!')
        assert np.allclose(m_red.faces, m_black.faces), 'Mesh Faces Index mismatch'
        self.mesh = m_red
        if not m_red_n and m_black_n:
            print('B-N>R', end=' ')
            self.mesh = m_black
            m_red_visual = m_red.visual.copy()
            m_red = m_black.copy()
            m_red_visual.mesh = m_red
            m_red.visual = m_red_visual

        red_mesh = PreCalcMesh(m_red, device, **LabeledColor.Red._asdict())
        black_mesh = NearestMeshQuery(m_black, device, **LabeledColor.Black._asdict())
        self.red_mesh = red_mesh if red_mesh else None
        self.black_mesh = black_mesh if black_mesh else None
        self.scan_info = scan_info

        self.is_perfect_A = False
        self.is_perfect_B = False
        self.is_good_A = False
        if not m_red.is_watertight or not m_black.is_watertight:
            print('NW!', end='!')
        if scan_info.PerfectA:
            print('PerfectA', end=' ' if black_mesh else '+')
            self.is_perfect_A = True
        elif scan_info.Trusted:
            self.is_good_A = True
        if scan_info.PerfectB:
            print('PerfectB', end=' ' if red_mesh else '+')
            self.is_perfect_B = True
        if scan_info.Trusted:
            print('Trusted', end=' ' if red_mesh else '!')
            if not red_mesh:
                self.is_perfect_B = True

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

    def query_pts_summary(self):
        if self.scan_info.PerfectB or (self.red_mesh is None and self.scan_info.Trusted):
            red_area = self.mesh.area
        elif self.red_mesh is not None:
            red_area = self.red_mesh.get_labeled_area()
        else:
            raise RuntimeError('Untrusted mesh without Red Label!')

        black_area = 0.0
        if not self.scan_info.PerfectA:
            if self.black_mesh is not None:
                black_area = self.black_mesh.get_labeled_area()
            else:
                raise RuntimeError('Imperfect mesh without Black Label!')

        return red_area, black_area, self.mesh.area


def main(args):
    args.red_path = args.data_dir / 'red'
    args.black_path = args.data_dir / 'black'
    gen_path = args.data_dir / 'gen'

    register = MeshRegister(args)

    black_files = set(os.fspath(a.relative_to(args.black_path)) for a in args.black_path.glob('scan_*/*_output.ply'))
    red_files = set(os.fspath(a.relative_to(args.red_path)) for a in args.red_path.glob('scan_*/*_output.ply'))
    all_files = list(red_files.intersection(black_files))
    all_files.sort(key=lambda x: (int(os.path.dirname(x).split('_')[-1]), int(os.path.basename(x).split('_')[0])))

    err = 0
    consistency = 0
    unknown = 0
    valid = 0
    red_perc = 0.0
    black_perc = 0.0

    perfect_A = 0
    good_A = 0
    perfect_B = 0

    for file in all_files:
        try:
            scan_info = register.check_scan(os.path.dirname(file), os.path.basename(file))
            if not scan_info.Usable:
                print(file, '-->Unusable Skip')
                continue
            print(file, end=' ')
            m = AnnotatedMeshSummary(args, file, scan_info)
            if m.is_perfect_A:
                perfect_A += 1
            elif m.is_good_A:
                good_A += 1
            if m.is_perfect_B:
                perfect_B += 1
            red_area, black_area, mesh_area = m.query_pts_summary()
            red_area /= mesh_area
            black_area /= mesh_area
            red_perc += red_area
            black_perc += black_area
            print(f'OK, {red_area=}, {black_area=}')
            valid += 1
        except RuntimeError as e:
            print(colored('RuntimeError: ', 'magenta'), e)
            err += 1
        except AssertionError as e:
            print(colored('AssertionError: ', 'magenta'), e)
            consistency += 1
        except Exception as e:
            print(colored('UnknownError: ', 'magenta'), e)
            unknown += 1

    print(f'{err=}, {consistency=}, {unknown=}, {valid=}, total={len(all_files)}')
    print(f'{perfect_A=}, {good_A=}, {perfect_B=}')
    print(f'red_area={red_perc / valid}, black_area={black_perc / valid}')
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
