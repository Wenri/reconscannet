import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import mcubes
import numpy as np
import open3d as o3d
import torch
import trimesh
from torch.utils.data import DataLoader

from configs.config_utils import CONFIG
from configs.scannet_config import ScannetConfig
from dataloader import ISCNet_ScanNet, collate_fn, my_worker_init_fn
from net_utils.box_util import get_3d_box_cuda
from net_utils.libs import flip_axis_to_camera_cuda, flip_axis_to_depth_cuda
from net_utils.utils import initiate_environment
from net_utils.voxel_util import voxels_from_scannet
from scannet.scannet_utils import chair_cat
from scannet.visualization.vis_for_demo import Vis_base


def pcd_to_mesh(xyz, output_file):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcd.estimate_normals()
    output_file_pcd = output_file.with_suffix('.pcd')
    o3d.io.write_point_cloud(os.fspath(output_file_pcd), pcd)

    poisson_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1,
                                                                                linear_fit=False)
    output_file_fn = output_file.with_suffix('.ply')
    o3d.io.write_triangle_mesh(os.fspath(output_file_fn), poisson_mesh)
    return output_file_fn


def tri_to_mesh(tri, output_file):
    output_file_fn = output_file.with_suffix('.ply')
    tri.export(os.fspath(output_file_fn))
    return output_file_fn


def vox_to_mesh(occ_hat, output_file, threshold=0.5, padding=0.1):
    """ Extracts the mesh from the predicted occupancy grid.

    Args:
        occ_hat (tensor): value grid of occupancies
        z (tensor): latent code z
        c (tensor): latent conditioned code c
    """
    # Some short hands
    n_x, n_y, n_z = occ_hat.shape
    box_size = 1 + padding

    # Make sure that mesh is watertight
    occ_hat_padded = np.pad(
        occ_hat, 1, 'constant', constant_values=-1e6)
    vertices, triangles = mcubes.marching_cubes(
        occ_hat_padded, threshold)
    # Strange behaviour in libmcubes: vertices are shifted by 0.5
    vertices -= 0.5
    # Undo padding
    vertices -= 1
    # Normalize to bounding box
    vertices /= np.array([n_x - 1, n_y - 1, n_z - 1])
    vertices = box_size * (vertices - 0.5)

    # Create mesh
    mesh = trimesh.Trimesh(vertices, triangles, process=False)

    return tri_to_mesh(mesh, output_file)


def get_bbox(dataset_config, center_label, heading_class_label, heading_residual_label,
             size_class_label, size_residual_label, **kwargs):
    centers_upright_camera = flip_axis_to_camera_cuda(center_label)
    heading_angles = dataset_config.class2angle_cuda(heading_class_label, heading_residual_label)

    # gather proposal box size
    box_size = dataset_config.class2size_cuda(size_class_label, size_residual_label)

    corners_3d_upright_camera = get_3d_box_cuda(box_size, -heading_angles, centers_upright_camera)

    kwargs['boxes3d'] = flip_axis_to_depth_cuda(corners_3d_upright_camera)
    kwargs['box_centers'] = center_label
    kwargs['box_sizes'] = box_size
    kwargs['heading_angles'] = heading_angles

    cos_orientation, sin_orientation = torch.cos(heading_angles), torch.sin(heading_angles)
    zero_orientation, one_orientation = torch.zeros_like(heading_angles), torch.ones_like(heading_angles)
    axis_rectified = torch.stack([torch.stack([cos_orientation, -sin_orientation, zero_orientation], dim=-1),
                                  torch.stack([sin_orientation, cos_orientation, zero_orientation], dim=-1),
                                  torch.stack([zero_orientation, zero_orientation, one_orientation], dim=-1)], dim=-1)
    kwargs['axis_rectified'] = axis_rectified

    return kwargs


def run(opt, cfg):
    dataset = ISCNet_ScanNet(cfg, mode='test')
    dataloader = DataLoader(dataset=dataset,
                            num_workers=cfg.config['device']['num_workers'],
                            batch_size=1,
                            shuffle=False,
                            collate_fn=collate_fn,
                            worker_init_fn=my_worker_init_fn)

    for cur_iter, data in enumerate(dataloader):
        bid = 0
        c = SimpleNamespace(**{k: v[bid] for k, v in get_bbox(cfg.dataset_config, **data).items()})

        instance_models = []
        center_list = []
        vector_list = []

        out_scan_dir = opt.output_dir / f'scan_{c.scan_idx}'
        print(f'scan_{c.scan_idx}')

        for idx in c.box_label_mask.nonzero(as_tuple=True)[0]:
            if c.shapenet_catids[idx] not in chair_cat:
                continue

            out_scan_dir.mkdir(exist_ok=True)

            ins_id = c.object_instance_labels[idx]
            ins_pc = c.point_clouds[c.point_instance_labels == ins_id].cuda()

            voxels, pc_normed = voxels_from_scannet(ins_pc, c.box_centers[idx].cuda(), c.box_sizes[idx].cuda(),
                                                    c.axis_rectified[idx].cuda())

            np.savez(out_scan_dir / f"{idx}_{c.shapenet_ids[idx]}_input.npz", pc=ins_pc.cpu().numpy(),
                     pc_normed=pc_normed.cpu().numpy(),
                     voxels=voxels[0].cpu().numpy())

        if not instance_models:
            continue

        scene = Vis_base(scene_points=c.point_clouds, instance_models=instance_models, center_list=center_list,
                         vector_list=vector_list)
        camera_center = np.array([0, -3, 3])
        scene.visualize(centroid=camera_center, offline=True, save_path=out_scan_dir / 'pred.png')


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
