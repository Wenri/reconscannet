# Base data of networks
# author: ynie
# date: Feb, 2020
import os
from abc import ABC
from pathlib import Path

from torch.utils.data import Dataset

from utils.read_and_write import read_json


class ScanNet(Dataset, ABC):
    def __init__(self, cfg, mode):
        """
        initiate SUNRGBD dataset for data loading
        :param cfg: config file
        :param mode: train/val/test mode
        """
        self.config = cfg.config
        self.dataset_config = cfg.dataset_config
        self.mode = mode
        split_file = os.path.join(cfg.config['data']['split'], 'scannetv2_' + mode + '.json')
        self.split = [{k: Path(v).relative_to('datasets') for k, v in a.items()} for a in read_json(split_file)]

    def __len__(self):
        return len(self.split)
