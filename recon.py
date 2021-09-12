import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configs.config_utils import CONFIG
from configs.scannet_config import ScannetConfig
from dataloader import ISCNet_ScanNet, collate_fn, my_worker_init_fn
from msn.model import MSN
from msn.utils import weights_init
from net_utils.utils import initiate_environment


def run(opt):
    network = MSN(num_points=opt.num_points, n_primitives=opt.n_primitives)
    network.cuda()
    network.apply(weights_init)

    if opt.model != '':
        network.load_state_dict(torch.load(opt.model))
        print("Previous weight loaded ")

    network.eval()

    dataset = ISCNet_ScanNet(cfg, mode='test')
    dataloader = DataLoader(dataset=dataset,
                            num_workers=cfg.config['device']['num_workers'],
                            batch_size=1,
                            shuffle=False,
                            collate_fn=collate_fn,
                            worker_init_fn=my_worker_init_fn)

    for cur_iter, data in enumerate(dataloader):
        for idx in torch.nonzero(data['box_label_mask'].squeeze(0)):
            print(data)

        output1, output2, expansion_penalty = network(partial.transpose(2, 1).contiguous())
        print(data)


def parse_args():
    """PARAMETERS"""
    base_dir = Path(__file__).parent
    parser = argparse.ArgumentParser('Instance Scene Completion.')
    parser.add_argument('--config', type=Path, default=base_dir / 'configs' / 'config_files' / 'ISCNet_test.yaml',
                        help='configure file for training or testing.')
    parser.add_argument('--mode', type=str, default='train', help='train, test or demo.')
    parser.add_argument('--demo_path', type=Path, default=base_dir / 'demo' / 'inputs' / 'scene0549_00.off',
                        help='Please specify the demo path.')
    parser.add_argument('--num_points', type=int, default=8192, help='number of points')
    parser.add_argument('--n_primitives', type=int, default=16, help='number of primitives in the atlas')
    parser.add_argument('--model', type=Path, default=Path('trained_model', 'network.pth'),
                        help='optional reload model path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = CONFIG(args.config, dataset_config=ScannetConfig())

    initiate_environment(cfg.config)

    '''Configuration'''
    cfg.log_string('Loading configurations.')
    cfg.log_string(cfg.config)
    cfg.write_config()

    run(args)
