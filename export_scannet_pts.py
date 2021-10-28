import argparse
import itertools
import os
import struct
from pathlib import Path
from types import SimpleNamespace

import mcubes
import numpy as np
import open3d as o3d
import torch
import trimesh
from torch.utils.data import DataLoader

from configs.config_utils import CONFIG
from configs.path_config import PathConfig
from configs.scannet_config import ScannetConfig
from dataloader import ISCNet_ScanNet, collate_fn, my_worker_init_fn
from net_utils.box_util import get_3d_box_cuda
from net_utils.libs import flip_axis_to_camera_cuda, flip_axis_to_depth_cuda
from net_utils.utils import initiate_environment
from net_utils.voxel_util import voxels_from_scannet, points_from_scannet, roty
from scannet.scannet_utils import ShapeNetCat
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


def vox_to_mesh(occ_hat, output_file, threshold=0.5, padding=0.):
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
    # vertices -= 0.5
    # Undo padding
    vertices -= 1
    # Normalize to bounding box
    vertices /= np.array([n_x - 1, n_y - 1, n_z - 1])
    vertices = box_size * (vertices - 0.5)

    # Create mesh
    mesh = trimesh.Trimesh(vertices, triangles, process=False)

    return tri_to_mesh(mesh, output_file)


def write_pointcloud(filename, xyz_points, rgb_points=None):
    """ creates a .ply file of the point clouds generated
    """

    n_total, n_dim = xyz_points.shape
    assert n_dim == 3, 'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.asarray((255, 0, 0), dtype=np.uint8)
    if rgb_points.ndim < 2:
        rgb_points = np.broadcast_to(np.expand_dims(rgb_points, axis=0), shape=(n_total, 3))
    assert xyz_points.shape == rgb_points.shape, 'Input RGB colors should be Nx3 float array and have same size as ' \
                                                 'input XYZ points '

    # Write header of .ply file
    with open(filename, 'wb') as fid:
        fid.write(bytes('ply\n', 'utf-8'))
        fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
        fid.write(bytes('element vertex %d\n' % xyz_points.shape[0], 'utf-8'))
        fid.write(bytes('property float x\n', 'utf-8'))
        fid.write(bytes('property float y\n', 'utf-8'))
        fid.write(bytes('property float z\n', 'utf-8'))
        fid.write(bytes('property uchar red\n', 'utf-8'))
        fid.write(bytes('property uchar green\n', 'utf-8'))
        fid.write(bytes('property uchar blue\n', 'utf-8'))
        fid.write(bytes('end_header\n', 'utf-8'))

        # Write 3D points to .ply file
        for i in range(xyz_points.shape[0]):
            fid.write(bytearray(struct.pack("fffccc", xyz_points[i, 0], xyz_points[i, 1], xyz_points[i, 2],
                                            rgb_points[i, 0].tostring(), rgb_points[i, 1].tostring(),
                                            rgb_points[i, 2].tostring())))


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


def get_shapenet_obj_mesh(path_config, cat_id, model_id):
    obj_path = path_config.ShapeNetv2_path / cat_id / model_id / 'models' / 'model_normalized.obj'
    assert obj_path.exists()

    obj_mesh = trimesh.load(obj_path, force='mesh')
    obj_mesh.vertices = obj_mesh.vertices @ roty.T.numpy()

    obj_min, obj_max = np.min(obj_mesh.vertices, axis=0), np.max(obj_mesh.vertices, axis=0)
    obj_ctr = (obj_min + obj_max) / 2
    obj_mesh.vertices -= obj_ctr
    obj_scale = obj_max - obj_min
    # obj_boxsize = c.box_sizes[idx] @ transform_shapenet
    # obj_scale /= np.abs(obj_boxsize)
    obj_mesh.vertices /= obj_scale

    obj_min, obj_max = np.min(obj_mesh.vertices, axis=0), np.max(obj_mesh.vertices, axis=0)
    obj_mesh.vertices /= np.max(obj_max - obj_min)

    return obj_mesh


def get_scannet_mesh(path_config, scene_name):
    scene_folder = path_config.metadata_root / 'scans' / scene_name
    meta_file = scene_folder / (scene_name + '.txt')
    mesh_file = scene_folder / (scene_name + '_vh_clean.ply')

    with meta_file.open('r') as f:
        for line in f.readlines():
            if 'axisAlignment' in line:
                axis_align_matrix = np.asarray(line.rstrip().strip('axisAlignment = ').split(' '), dtype=np.float_)
                break
    axis_align_matrix.shape = (4, 4)

    mesh = trimesh.load_mesh(mesh_file)
    mesh.apply_transform(axis_align_matrix)

    return mesh


def run(opt, cfg):
    dataset = ISCNet_ScanNet(cfg, mode='test', split='train')
    dataloader = DataLoader(dataset=dataset,
                            num_workers=cfg.config['device']['num_workers'],
                            batch_size=1,
                            shuffle=False,
                            collate_fn=collate_fn,
                            worker_init_fn=my_worker_init_fn)
    path_config = PathConfig('scannet')
    cat_set = cfg.config['data']['classes']
    cat_set = getattr(ShapeNetCat, cat_set) if cat_set else None

    for cur_iter, data in enumerate(dataloader):
        # if cur_iter <= 931:
        #     continue

        bid = 0
        c = SimpleNamespace(**{k: v[bid] for k, v in get_bbox(cfg.dataset_config, **data).items()})

        instance_models = []
        center_list = []
        vector_list = []
        scan_idx = c.scan_idx.item()

        out_scan_dir = opt.output_dir / f'scan_{scan_idx}'
        print(f'scan_{scan_idx}')

        instance_indices = [idx for idx in c.box_label_mask.nonzero(as_tuple=True)[0]
                            if c.shapenet_catids[idx] in cat_set]

        if not instance_indices:
            continue

        scan_mesh = get_scannet_mesh(path_config, scene_name=dataset.split[scan_idx]['scan'].parent.name)
        scan_old_vert = torch.from_numpy(scan_mesh.vertices).float().cuda()

        for idx in instance_indices:
            out_scan_dir.mkdir(exist_ok=True)

            ins_id = c.object_instance_labels[idx]
            ins_pc = c.point_clouds[c.point_instance_labels == ins_id].cuda()

            scannet_transfer = (c.box_centers[idx].cuda(), c.box_sizes[idx].cuda(), c.axis_rectified[idx].cuda())
            voxels, pc_normed, overscan = voxels_from_scannet(ins_pc, *scannet_transfer)

            np.savez(out_scan_dir / f"{idx}_{c.shapenet_ids[idx]}_input.npz", pc=ins_pc.cpu().numpy(),
                     pc_normed=pc_normed[0].cpu().numpy(),
                     voxels=voxels[0].cpu().numpy(),
                     overscan=overscan.cpu().numpy())

            write_pointcloud(out_scan_dir / f"{idx}_{c.shapenet_ids[idx]}_partial_pc.ply",
                             (pc_normed[0] / overscan).cpu().numpy())

            obj_mesh = get_shapenet_obj_mesh(path_config, c.shapenet_catids[idx], c.shapenet_ids[idx])
            tri_to_mesh(obj_mesh, out_scan_dir / f"{idx}_{c.shapenet_ids[idx]}_scan2cad.ply")

            scan_pts = points_from_scannet(scan_old_vert, *scannet_transfer) / overscan

            for bound in itertools.count(start=0.7, step=0.1):
                scan_vert_mask = torch.all(torch.abs(scan_pts) <= bound, dim=1)
                scan_face_mask = torch.all(scan_vert_mask[scan_mesh.faces], dim=1)
                if torch.any(scan_face_mask):
                    scan_mesh.vertices = scan_pts.cpu().numpy()
                    scan_submesh, = scan_mesh.submesh(np.nonzero(scan_face_mask.cpu().numpy()))
                    tri_to_mesh(scan_submesh, out_scan_dir / f"{idx}_{c.shapenet_ids[idx]}_scan.ply")
                    break

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
