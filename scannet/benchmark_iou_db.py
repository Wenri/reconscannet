import argparse
import itertools
import os
import statistics
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import trimesh
from tqdm import tqdm

from external.common import compute_iou
from if_net.data_processing.implicit_waterproofing import implicit_waterproofing
from scannet.ABNormalDataset import ABNormalDataset


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
        scan_id, obj_id = int(f.parent.name.split('_')[1]), int(f.name.split('_')[0])
        npz_file = self.abnormal_dataset.load_by_id(scan_id, obj_id)
        if npz_file is None:
            return None
        npz_file, cls_id = npz_file

        try:
            m = trimesh.load(f, force='mesh')
        except Exception:
            return None

        pts = npz_file['pts']
        pts_mask = npz_file['pts_mask']
        occ_list, holes_list = implicit_waterproofing(m, pts)
        if np.any(holes_list):
            print('holes_list not empty', file=sys.stderr)
        iou = compute_iou(occ_list[pts_mask[:, 1]], pts_mask[pts_mask[:, 1], 0])
        return scan_id, obj_id, cls_id, iou


def main(args):
    bm = Benchmark()
    file_to_eval = sorted(Path(args.data_dir).glob('**/*_output.ply'))
    iou_scan = defaultdict(dict)
    with Pool(os.cpu_count()) as p:
        for scan_id, obj_id, cls_id, iou in filter(None, p.imap_unordered(bm.calc_f, tqdm(file_to_eval))):
            best_iou = iou_scan[scan_id].get(obj_id)
            if best_iou is None or iou > best_iou[0]:
                # print(scan_id, obj_id, iou)
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
