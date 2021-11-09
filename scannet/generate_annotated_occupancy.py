import argparse
import csv
import os
import sys
from collections import namedtuple
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import trimesh
from trimesh.repair import fix_normals

from export_scannet_pts import write_pointcloud
from external.libmesh import check_mesh_contains
from if_net.data_processing.implicit_waterproofing import create_grid_points_from_bounds
from utils.raytracing import PreCalcMesh, NearestMeshQuery

COLOR_BOUND = namedtuple('COLOR_BOUND', ('color_low', 'color_high'))
LabeledColor = namedtuple('LABELED_COLOR', ('Red', 'Black'))(
    Red=COLOR_BOUND((180, 0, 0, 0), (255, 50, 50, 255)),
    Black=COLOR_BOUND((0, 0, 0, 0), (50, 50, 50, 255))
)

RegisterFieldsA = namedtuple('RegisterFieldsA', ('Bad', 'Good', 'Fine', 'Fail'))
RegisterFieldsB = namedtuple('RegisterFieldsB', ('Good', 'Fail'))
RegisterSummary = namedtuple('RegisterSummary', ('Perfect', 'Trusted', 'Usable'), defaults=(False,) * 3)


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
        self.red_mesh = red_mesh if red_mesh else None
        self.black_mesh = black_mesh if black_mesh else None
        self.scan_info = scan_info

    def load_mesh(self, file_path):
        ml = [m for m in trimesh.load(file_path, process=False).split(only_watertight=False) if len(m.faces) > 10]
        ml = trimesh.util.concatenate(ml)
        if not ml.fill_holes():
            return None
        fix_normals(ml)
        return ml

    def query_pts(self, pts):
        contains = check_mesh_contains(self.red_mesh.mesh, pts.cpu().numpy())
        contains_pos = torch.from_numpy(contains).to(device=self.device)
        contains_ok = torch.from_numpy(contains).to(device=self.device)
        contains_neg = torch.logical_not(contains_pos)
        to_red_pts = pts[contains_neg]
        to_black_pts = pts[contains_pos]
        if len(to_red_pts):
            if self.red_mesh is not None:
                to_red_mask = self.red_mesh.check_is_verified(to_red_pts)
                to_red_rev = torch.logical_not(to_red_mask)
                to_red_mask[to_red_rev] = self.red_mesh.check_is_edge(to_red_pts[to_red_rev])
            elif self.scan_info.Trusted:
                to_red_mask = torch.ones(to_red_pts.shape[0], dtype=torch.bool, device=self.device)
            else:
                raise ValueError('Untrusted mesh without Red Label!')
            contains_ok[contains_neg] = to_red_mask
        if not self.scan_info.Perfect and len(to_black_pts):
            if self.black_mesh is not None:
                to_black_mask = torch.from_numpy(self.black_mesh.check_is_black(to_black_pts)).to(device=self.device)
            else:
                raise ValueError('Imperfect mesh without Black Label!')
            contains_pos[contains] ^= to_black_mask
        return torch.stack((contains_pos, contains_ok), dim=-1)


class MeshRegister:
    CSV_TITLE_1 = ('标注方式1', '标注方式2')
    CSV_TITLE_2 = ('很差（目测60%以上都要涂黑）：', '很好（轮廓完整，且无冗余，不需要涂黑）',
                   '部分涂黑之后，可以达到“很好”', '重建失败（文件打开后没有点）',
                   '很好（不需要涂红）', '重建失败（文件打开后没有点）')

    def __init__(self, args):
        filesA_len = len(RegisterFieldsA._fields)
        registerA = {}
        registerB = {}
        header_seen = 0
        with args.csv_file.open() as f:
            for row in csv.reader(f):
                if not header_seen:
                    assert row[1] == self.CSV_TITLE_1[0] and row[filesA_len + 1] == self.CSV_TITLE_1[1]
                    header_seen += 1
                    continue
                elif header_seen == 1:
                    assert tuple(row[1:]) == self.CSV_TITLE_2
                    header_seen += 1
                    continue
                scan_name = tuple(row[0].split('_')[:2]) if row[0] else None
                assert scan_name not in registerA and scan_name not in registerB
                registerA[scan_name] = RegisterFieldsA._make(bool(a.strip()) for a in row[1:filesA_len + 1])
                registerB[scan_name] = RegisterFieldsB._make(bool(a.strip()) for a in row[filesA_len + 1:])

        self.registerA = registerA
        self.registerB = registerB

    def check_scan(self, scan_name, instance_id):
        scan_key = (scan_name.replace('_', ''), instance_id.split('_')[0])
        a = self.registerA[scan_key]
        b = self.registerB[scan_key]
        if a.Bad or a.Fail or b.Fail:
            return RegisterSummary()
        return RegisterSummary(Usable=True, Perfect=a.Good, Trusted=a.Good or a.Fine or b.Good)


def main(args):
    device = torch.device('cuda')
    pool = Pool()

    args.red_path = args.data_dir / 'red'
    args.black_path = args.data_dir / 'black'
    gen_path = args.data_dir / 'gen'

    register = MeshRegister(args)

    black_files = set(os.fspath(a.relative_to(args.black_path)) for a in args.black_path.glob('scan_*/*_output.ply'))
    red_files = set(os.fspath(a.relative_to(args.red_path)) for a in args.red_path.glob('scan_*/*_output.ply'))
    all_files = red_files.intersection(black_files)

    pts = torch.from_numpy(create_grid_points_from_bounds(-0.55, .55, 64)).to(device=device, dtype=torch.float)
    pts_split = torch.tensor_split(pts, 32)

    err = 0
    consistency = 0
    for file in all_files:
        try:
            scan_info = register.check_scan(os.path.dirname(file), os.path.basename(file))
            if not scan_info.Usable:
                continue
            m = AnnotatedMesh(args, file, scan_info, device, pool=pool)
            pts_mask = torch.cat([m.query_pts(p) for p in pts_split])
            inpts = pts[torch.all(pts_mask, dim=-1)]
            outpts = pts[~pts_mask[:, 0] & pts_mask[:, 1]]
            mesh_file = gen_path / file
            mesh_file.parent.mkdir(exist_ok=True)
            write_pointcloud(mesh_file.with_suffix('.in.ply'), inpts.cpu().numpy())
            write_pointcloud(mesh_file.with_suffix('.out.ply'), outpts.cpu().numpy(),
                             rgb_points=np.asarray((0, 0, 255), dtype=np.uint8))
            print(file, ' OK')
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
        'data', 'anno1', 'table_ours'))
    parser.add_argument('--csv_file', type=Path, help='optional reload model path', default=Path(
        'data', 'anno1', 'table_ours', 'table.csv'))
    return parser.parse_args()


if __name__ == '__main__':
    sys.exit(main(parse_args()))
