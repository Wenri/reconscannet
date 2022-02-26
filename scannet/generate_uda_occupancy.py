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

COLOR_BOUND = namedtuple('COLOR_BOUND', ('color_low', 'color_high'))
LabeledColor = namedtuple('LABELED_COLOR', ('Red', 'Black'))(
    Red=COLOR_BOUND((150, 0, 0, 0), (255, 150, 150, 255)),
    Black=COLOR_BOUND((0, 0, 0, 0), (100, 100, 100, 255))
)


class AnnotatedMesh:
    def __init__(self, args, mesh_file, device, **kwargs):
        self.device = device
        self.mesh, _, _ = self.load_mesh(args.data_dir / mesh_file)
        if not self.mesh.is_watertight:
            print('NW!', end='!')

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

        return torch.stack((contains_pos, contains_ok), dim=-1)


def main(args):
    device = torch.device('cuda')
    pool = Pool()

    gen_path = args.data_dir / 'gen'

    all_files = [os.fspath(a.relative_to(args.data_dir)) for a in args.data_dir.glob('scan_*/*_output.ply')]
    all_files.sort(key=lambda x: (int(os.path.dirname(x).split('_')[-1]), int(os.path.basename(x).split('_')[0])))

    pts = torch.from_numpy(create_grid_points_from_bounds(-0.55, .55, 64)).to(device=device, dtype=torch.float)
    pts_split = torch.tensor_split(pts, 64)

    err = 0
    consistency = 0
    unknown = 0
    for idx, file in enumerate(all_files):
        try:
            print(file, end=' ')
            m = AnnotatedMesh(args, file, device, pool=pool)
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
    parser.add_argument('--data_dir', type=Path, help='optional reload model path', default=Path('out_recon_uda_ext_cls'))
    parser.add_argument('--start_from', type=int, help='starting from scan id', default=0)
    return parser.parse_args()


if __name__ == '__main__':
    sys.exit(main(parse_args()))
