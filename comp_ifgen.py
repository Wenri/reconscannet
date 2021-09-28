import argparse
from pathlib import Path

import numpy as np
import torch

from configs.config_utils import CONFIG
from configs.scannet_config import ScannetConfig
from export_scannet_pts import pcd_to_mesh, tri_to_mesh, vox_to_mesh
from if_net.models.generation import Generator
from if_net.models.local_model import ShapeNetPoints
from net_utils.utils import initiate_environment
from net_utils.voxel_util import pointcloud2voxel_fast


def run(opt, cfg):
    network = ShapeNetPoints()

    generator = Generator(network, 0.45, 'sample_input', checkpoint=opt.model, resolution=opt.retrieval_res,
                          batch_points=opt.batch_points)

    network.cuda()
    network.eval()

    npzf = np.load('sample_input/1007e20d5e811b308351982a6e40cf41/voxelized_point_cloud_32res_300points.npz')
    occ = np.unpackbits(npzf['compressed_occupancies'])
    voxels = np.reshape(occ, (npzf['res'],) * 3)
    npzf = npzf['point_cloud']

    out_scan_dir = opt.output_dir / 'sample_input'
    out_scan_dir.mkdir(exist_ok=True)

    pcd_to_mesh(npzf, out_scan_dir / 'voxelized_point_cloud')

    voxels = torch.from_numpy(voxels).float()

    vox_to_mesh(voxels.numpy(), out_scan_dir / 'voxelized_mesh')

    npzf = torch.from_numpy(npzf).float().cuda().unsqueeze(0)
    all_voxels = pointcloud2voxel_fast(npzf)

    vox_to_mesh(all_voxels[0].cpu().numpy(), out_scan_dir / 'all_voxelized_mesh', threshold=0.0)

    voxels = voxels.cuda().unsqueeze(0)
    logits = generator.generate_mesh(voxels)
    meshes = generator.mesh_from_logits(logits)

    output_pcd_fn = tri_to_mesh(meshes, out_scan_dir / 'joutout')

    logits = generator.generate_mesh(all_voxels)
    meshes = generator.mesh_from_logits(logits)

    output_pcd_fn = tri_to_mesh(meshes, out_scan_dir / 'alloutout')


def parse_args():
    """PARAMETERS"""
    base_dir = Path(__file__).parent
    parser = argparse.ArgumentParser('Instance Scene Completion.')
    parser.add_argument('--config', type=Path, default=base_dir / 'configs' / 'config_files' / 'ISCNet_test.yaml',
                        help='configure file for training or testing.')
    parser.add_argument('--mode', type=str, default='train', help='train, test or demo.')
    parser.add_argument('--demo_path', type=Path, default=base_dir / 'demo' / 'inputs' / 'scene0549_00.off',
                        help='Please specify the demo path.')
    parser.add_argument('--res', default=32, type=int)
    parser.add_argument('--retrieval_res', default=256, type=int)
    parser.add_argument('--model', type=Path, default=Path('trained_model', 'checkpoint_epoch_999.tar'),
                        help='optional reload model path')
    parser.add_argument('--batch_points', default=100000, type=int)
    parser.add_argument('--output_dir', type=Path, default=Path('out'),
                        help='output path')
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
