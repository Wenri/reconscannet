"""
Path configs for the whole project.
author: ynie
date: July, 2020
"""
import os
import pickle
from itertools import chain
from pathlib import PurePath, Path

import numpy as np

from scannet.scannet_utils import read_label_mapping

SHAPENETCLASSES = ['void',
                   'table', 'jar', 'skateboard', 'car', 'bottle',
                   'tower', 'chair', 'bookshelf', 'camera', 'airplane',
                   'laptop', 'basket', 'sofa', 'knife', 'can',
                   'rifle', 'train', 'pillow', 'lamp', 'trash_bin',
                   'mailbox', 'watercraft', 'motorbike', 'dishwasher', 'bench',
                   'pistol', 'rocket', 'loudspeaker', 'file cabinet', 'bag',
                   'cabinet', 'bed', 'birdhouse', 'display', 'piano',
                   'earphone', 'telephone', 'stove', 'microphone', 'bus',
                   'mug', 'remote', 'bathtub', 'bowl', 'keyboard',
                   'guitar', 'washer', 'bicycle', 'faucet', 'printer',
                   'cap', 'clock', 'helmet', 'flowerpot', 'microwaves']

ScanNet_OBJ_CLASS_IDS = np.array([1, 7, 8, 13, 20, 31, 34, 43])
ShapeNetIDMap = {'4379243': 'table', '3593526': 'jar', '4225987': 'skateboard', '2958343': 'car', '2876657': 'bottle',
                 '4460130': 'tower', '3001627': 'chair', '2871439': 'bookshelf', '2942699': 'camera',
                 '2691156': 'airplane', '3642806': 'laptop', '2801938': 'basket', '4256520': 'sofa', '3624134': 'knife',
                 '2946921': 'can', '4090263': 'rifle', '4468005': 'train', '3938244': 'pillow', '3636649': 'lamp',
                 '2747177': 'trash_bin', '3710193': 'mailbox', '4530566': 'watercraft', '3790512': 'motorbike',
                 '3207941': 'dishwasher', '2828884': 'bench', '3948459': 'pistol', '4099429': 'rocket',
                 '3691459': 'loudspeaker', '3337140': 'file cabinet', '2773838': 'bag', '2933112': 'cabinet',
                 '2818832': 'bed', '2843684': 'birdhouse', '3211117': 'display', '3928116': 'piano',
                 '3261776': 'earphone', '4401088': 'telephone', '4330267': 'stove', '3759954': 'microphone',
                 '2924116': 'bus', '3797390': 'mug', '4074963': 'remote', '2808440': 'bathtub', '2880940': 'bowl',
                 '3085013': 'keyboard', '3467517': 'guitar', '4554684': 'washer', '2834778': 'bicycle',
                 '3325088': 'faucet', '4004475': 'printer', '2954340': 'cap', '3046257': 'clock', '3513137': 'helmet',
                 '3991062': 'flowerpot', '3761084': 'microwaves'}


class PathConfig(object):
    def __init__(self, dataset, prefix=PurePath()):
        self.ShapeNetv2_path = Path(prefix, 'ShapeNetCore.v2')
        if dataset == 'scannet':
            self.metadata_root = Path(prefix, 'scannet')
            self.split_files = [Path(prefix, 'splits', 'scannet', 'scannetv2_train.txt'),
                                Path(prefix, 'splits', 'scannet', 'scannetv2_val.txt')]
            all_scenes = set(chain.from_iterable(np.loadtxt(split_file, dtype=str) for split_file in self.split_files))

            scene_paths = (self.metadata_root / 'scans').iterdir()
            self.scene_paths = [scene_path for scene_path in scene_paths if os.path.basename(scene_path) in all_scenes]
            self.scan2cad_annotation_path = self.metadata_root / 'scan2cad_download_link' / 'full_annotations.json'
            self.processed_data_path = self.metadata_root / 'processed_data'

            label_type = 'synsetoffset'
            self.raw_label_map_file = self.metadata_root / ('raw_lablemap_to_' + label_type + '.pkl')
            self.OBJ_CLASS_IDS = ScanNet_OBJ_CLASS_IDS

            if not self.raw_label_map_file.exists():
                LABEL_MAP_FILE = self.metadata_root / 'scannetv2-labels.combined.tsv'
                assert LABEL_MAP_FILE.exists()
                raw_label_map = read_label_mapping(LABEL_MAP_FILE, label_from='raw_category', label_to=label_type)
                label_map = {key: SHAPENETCLASSES.index(ShapeNetIDMap[item]) if item in ShapeNetIDMap else 0
                             for key, item in raw_label_map.items()}

                with self.raw_label_map_file.open('wb') as file:
                    pickle.dump(label_map, file)

            self.processed_data_path.mkdir(exist_ok=True)


if __name__ == '__main__':
    path_config = PathConfig('scannet')
    print(path_config.scene_paths)
