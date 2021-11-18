import argparse
import itertools
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh

from external.common import compute_iou
from if_net.data_processing.implicit_waterproofing import implicit_waterproofing
from scannet.ABNormalDataset import ABNormalDataset

ABCfg = {'data': {
    'abnormal': 'data/anno1',
    'path': 'data/ShapeNet',
    'pointcloud_n': 2688,
    'points_subsample': 5000
}}


def main(args):
    abnormal_dataset = ABNormalDataset('train', ABCfg)
    file_to_eval = Path(args.data_dir).glob('*/*_mesh.ply')
    iou_scan = defaultdict(dict)
    for f in file_to_eval:
        scan_id = f.parent.name
        obj_id = f.name.split('_')[0]
        npz_file = abnormal_dataset.load_by_id(scan_id, obj_id)
        if npz_file is None:
            continue
        npz_file, cls_id = npz_file
        m = trimesh.load(f, force='mesh')
        pts = npz_file['pts']
        pts_mask = npz_file['pts_mask']
        occ_list, holes_list = implicit_waterproofing(m, pts)
        if np.any(holes_list):
            raise RuntimeError('holes_list not empty')
        iou = compute_iou(occ_list[pts_mask[:, 1]], pts_mask[pts_mask[:, 1], 0])
        best_iou = iou_scan[scan_id].get(obj_id)
        if best_iou is None or iou > best_iou[0]:
            print(scan_id, obj_id, iou)
            iou_scan[scan_id][obj_id] = (iou, cls_id)

    iou_cls = defaultdict(list)
    for iou, c in itertools.chain.from_iterable(a.values() for a in iou_scan.values()):
        iou_cls[c].append(iou)

    for c, iou_list in iou_cls.items():
        total_iou = statistics.mean(iou_list)
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
