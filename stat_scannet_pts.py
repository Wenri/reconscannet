import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config_utils import CONFIG
from configs.scannet_config import ScannetConfig
from dataloader import ISCNet_ScanNet, collate_fn, my_worker_init_fn
from export_scannet_pts import get_bbox
from net_utils.utils import initiate_environment


def run(opt, cfg):
    dataset = ISCNet_ScanNet(cfg, mode='test', split='train')
    dataloader = DataLoader(dataset=dataset,
                            num_workers=cfg.config['device']['num_workers'],
                            batch_size=1,
                            shuffle=False,
                            collate_fn=collate_fn,
                            worker_init_fn=my_worker_init_fn)

    stat_cat_dict = defaultdict(int)
    metadata_file = os.path.join(cfg.config['data']['path'], 'metadata.yaml')
    with open(metadata_file, 'r') as f:
        metadata = yaml.load(f)

    for data in tqdm(dataloader, file=sys.stdout, dynamic_ncols=True):
        bid = 0
        c = SimpleNamespace(**{k: v[bid] for k, v in get_bbox(cfg.dataset_config, **data).items()})

        scan_idx = c.scan_idx.item()

        out_scan_dir = opt.output_dir / f'scan_{scan_idx}'

        instance_indices = [idx for idx in c.box_label_mask.nonzero(as_tuple=True)[0]]

        if not instance_indices:
            continue

        for idx in instance_indices:
            out_scan_dir.mkdir(exist_ok=True)

            shapenatcat_id = c.shapenet_catids[idx]
            stat_cat_dict[shapenatcat_id] += 1

    for id, cnt in stat_cat_dict.items():
        print(f'{metadata[id]["name"]}: {cnt}')


def parse_args():
    """PARAMETERS"""
    base_dir = Path(__file__).parent
    parser = argparse.ArgumentParser('Instance Scene Completion.')
    parser.add_argument('--config', type=Path, default=base_dir / 'configs' / 'config_files' / 'if_net_test.yaml',
                        help='configure file for training or testing.')
    parser.add_argument('--mode', type=str, default='train', help='train, test or demo.')
    parser.add_argument('--demo_path', type=Path, default=base_dir / 'demo' / 'inputs' / 'scene0549_00.off',
                        help='Please specify the demo path.')
    parser.add_argument('--num_points', type=int, default=8192, help='number of points')
    parser.add_argument('--n_primitives', type=int, default=16, help='number of primitives in the atlas')
    parser.add_argument('--model', type=Path, default=Path('trained_model', 'checkpoint_epoch_999.tar'),
                        help='optional reload model path')
    parser.add_argument('--output_dir', type=Path, default=Path('out'),
                        help='output path')
    parser.add_argument('--res', default=32, type=int)
    parser.add_argument('--retrieval_res', default=256, type=int)
    parser.add_argument('--batch_points', default=100000, type=int)
    return parser.parse_args()


def main(args):
    cfg = CONFIG(args.config, dataset_config=ScannetConfig())

    initiate_environment(cfg.config)

    '''Configuration'''
    cfg.log_string('Loading configurations.')
    cfg.log_string(cfg.config)
    cfg.write_config()

    run(args, cfg)


if __name__ == '__main__':
    main(parse_args())
