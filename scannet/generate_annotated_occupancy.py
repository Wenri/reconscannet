import argparse
import os
import sys
from collections import namedtuple
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import trimesh
from trimesh.repair import fix_normals

from utils.raytracing import PreCalcMesh

COLOR_BOUND = namedtuple('COLOR_BOUND', ('color_low', 'color_high'))
LabeledColor = namedtuple('LABELED_COLOR', ('Red', 'Black'))(
    Red=COLOR_BOUND((180, 0, 0, 0), (255, 50, 50, 255)),
    Black=COLOR_BOUND((0, 0, 0, 0), (50, 50, 50, 255))
)



class AnnotatedMesh:
    def __init__(self, args, mesh_file, device, **kwargs):
        m_red = self.load_mesh(args.red_path / mesh_file)
        m_black = self.load_mesh(args.black_path / mesh_file)
        if m_red is None or m_black is None:
            raise RuntimeError('load failed, maybe not watertight!')
        assert np.allclose(m_red.vertices, m_black.vertices)
        assert np.allclose(m_red.faces, m_black.faces)
        self.redmesh = PreCalcMesh(m_red, device, **LabeledColor.Red._asdict(), **kwargs)

    def load_mesh(self, file_path):
        ml = [m for m in trimesh.load(file_path, process=False).split(only_watertight=False) if len(m.faces) > 10]
        ml = trimesh.util.concatenate(ml)
        if not ml.fill_holes():
            return None
        fix_normals(ml)
        return ml


def main(args):
    device = torch.device('cuda')
    pool = Pool()

    args.red_path = args.data_dir / 'red'
    args.black_path = args.data_dir / 'black'

    black_files = set(os.fspath(a.relative_to(args.black_path)) for a in args.black_path.glob('scan_*/*_output.ply'))
    red_files = set(os.fspath(a.relative_to(args.red_path)) for a in args.red_path.glob('scan_*/*_output.ply'))
    all_files = red_files.intersection(black_files)

    err = 0
    consistency = 0
    for file in all_files:
        try:
            m = AnnotatedMesh(args, file, device, pool=pool)
        except RuntimeError as e:
            err += 1
        except AssertionError as e:
            consistency += 1

    print(err, consistency, len(all_files))
    return os.EX_OK


def parse_args():
    """PARAMETERS"""
    base_dir = Path(__file__).parent
    parser = argparse.ArgumentParser('Instance Scene Completion.')
    parser.add_argument('--max_samples', type=int, default=60000, help='number of points')
    parser.add_argument('--data_dir', type=Path, help='optional reload model path', default=Path(
        'data', 'anno1', 'chair'))
    return parser.parse_args()


if __name__ == '__main__':
    sys.exit(main(parse_args()))
