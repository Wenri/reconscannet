import argparse
from pathlib import Path
from types import SimpleNamespace

from torch.utils.data import DataLoader

from configs.config_utils import CONFIG
from configs.scannet_config import ScannetConfig
from dataloader import ISCNet_ScanNet, collate_fn, my_worker_init_fn
from export_scannet_pts import get_bbox
from net_utils.utils import initiate_environment
from net_utils.voxel_util import voxels_from_scannet


def run(opt, cfg):
    dataset = ISCNet_ScanNet(cfg, mode='test', split='train')
    dataloader = DataLoader(dataset=dataset,
                            num_workers=cfg.config['device']['num_workers'],
                            batch_size=1,
                            shuffle=False,
                            collate_fn=collate_fn,
                            worker_init_fn=my_worker_init_fn)

    f = open('/tmp/maptab.txt', 'w')
    for cur_iter, data in enumerate(dataloader):
        bid = 0
        c = SimpleNamespace(**{k: v[bid] for k, v in get_bbox(cfg.dataset_config, **data).items()})

        scan_idx = c.scan_idx.item()
        scene_name = dataset.split[scan_idx]['scan']

        for idx in c.box_label_mask.nonzero(as_tuple=True)[0]:
            ins_id = int(c.object_instance_labels[idx])
            print(f'scan{scan_idx}_{idx}_{c.shapenet_ids[idx][:8]}:{scene_name.parent.name}_insid_{ins_id}', end='_',
                  file=f)
            print('center:', c.box_centers[idx].tolist(), end=':', file=f)
            ins_pc = c.point_clouds[c.point_instance_labels == ins_id].cuda()

            scannet_transfer = (c.box_centers[idx].cuda(), c.box_sizes[idx].cuda(), c.axis_rectified[idx].cuda())
            voxels, pc_normed, overscan = voxels_from_scannet(ins_pc, *scannet_transfer)

            print('overscan_', overscan.item(), file=f)


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
    parser.add_argument('--model', type=Path, default=Path('trained_model', 'checkpoint_epoch_999.tar'),
                        help='optional reload model path')
    parser.add_argument('--output_dir', type=Path, default=Path('out_export'),
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
