import argparse
import itertools
import math
import os
import sys
from collections import defaultdict
from contextlib import closing
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import trimesh
from pytorch3d.ops import knn_points
from tqdm import tqdm
from trimesh.sample import sample_surface_even

from external.common import compute_iou
from if_net.data_processing.implicit_waterproofing import implicit_waterproofing
from scannet.ABNormalDataset import ABNormalDataset
from scannet.summary_anno import LabeledColor
from utils.raytracing import get_labeled_face


class Benchmark:
    ABCfg = {'data': {
        'abnormal': 'data/annoALL',
        'path': 'data/ShapeNet',
        'pointcloud_n': 2688,
        'points_subsample': 5000
    }}

    def __init__(self):
        self.abnormal_dataset = ABNormalDataset('train', self.ABCfg)

    def calc_f(self, f):
        try:
            m = trimesh.load(f, force='mesh')
        except Exception as exc:
            return None
        scan_id, obj_id = int(f.parent.name.split('_')[1]), int(f.name.split('_')[0])
        npz_file = self.abnormal_dataset.load_by_id(scan_id, obj_id)
        if npz_file is None:
            return None
        with closing(npz_file):
            npz_path = Path(npz_file.fid.name)
            pts = npz_file['pts']
            pts_mask = npz_file['pts_mask']
        cls_id = self.abnormal_dataset.parse_cls(npz_path)
        occ_list, holes_list = implicit_waterproofing(m, pts)
        if np.any(holes_list):
            print('holes_list not empty', file=sys.stderr)
        iou = compute_iou(occ_list[pts_mask[:, 1]], pts_mask[pts_mask[:, 1], 0])

        scan_dir = npz_path.parent
        gen_dir = scan_dir.parent
        red_mesh = gen_dir.parent / 'red' / scan_dir.name / npz_path.with_suffix('.ply').name
        mred = trimesh.load_mesh(red_mesh)
        device = torch.device('cpu')
        face_mask = get_labeled_face(mred, device, **LabeledColor.Red._asdict())
        mred_pts = torch.from_numpy(mred.vertices).to(dtype=torch.float)
        if face_mask.any():
            mred_pts = mred_pts[tuple(frozenset(mred.faces[face_mask].flat)), ...]
        sample_pts, _ = sample_surface_even(m, 2500)
        sample_pts = torch.from_numpy(sample_pts).to(dtype=torch.float)
        x_nn = knn_points(mred_pts.unsqueeze(0), sample_pts.unsqueeze(0), K=1, return_sorted=False).dists
        cd = math.sqrt(x_nn.mean().item())
        return scan_id, obj_id, cls_id, iou, cd


def main(args):
    bm = Benchmark()
    file_to_eval = sorted(Path(args.data_dir).glob('**/*_output.ply'))
    iou_scan = defaultdict(dict)
    with Pool(os.cpu_count()) as p:
        for scan_id, obj_id, cls_id, iou, cd in filter(None, p.imap_unordered(bm.calc_f, tqdm(file_to_eval))):
            best_iou = iou_scan[scan_id].get(obj_id)
            if best_iou is None or iou > best_iou[0]:
                # print(scan_id, obj_id, iou)
                iou_scan[scan_id][obj_id] = (iou, cd, cls_id)

    iou_cls = defaultdict(list)
    for iou, cd, c in itertools.chain.from_iterable(a.values() for a in iou_scan.values()):
        iou_cls[c].append((iou, cd))

    for c, iou_list in iou_cls.items():
        a = np.asarray(iou_list)
        total_iou = a.mean(axis=0)
        print(f'{c=} {total_iou=}')

    return os.EX_OK


def parse_args():
    """PARAMETERS"""
    base_dir = Path(__file__).parent
    parser = argparse.ArgumentParser('Instance Scene Completion.')
    parser.add_argument('--max_samples', type=int, default=60000, help='number of points')
    parser.add_argument('--data_dir', type=Path, help='optional reload model path', default=Path('RfDExport'))
    return parser.parse_args()


if __name__ == '__main__':
    sys.exit(main(parse_args()))
