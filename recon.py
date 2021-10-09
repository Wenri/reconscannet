import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import vtkmodules.all as vtk
from torch.utils.data import DataLoader
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from configs.config_utils import CONFIG
from configs.scannet_config import ScannetConfig
from dataloader import ISCNet_ScanNet, collate_fn, my_worker_init_fn
from export_scannet_pts import tri_to_mesh, vox_to_mesh, get_bbox
from external.libsimplify import simplify_mesh
from if_net.models.data.config import get_model
from if_net.models.generator import Generator3D
from net_utils.utils import initiate_environment
from net_utils.voxel_util import voxels_from_scannet, transform_shapenet
from scannet.scannet_utils import chair_cat
from scannet.visualization.vis_for_demo import Vis_base
from utils.checkpoints import CheckpointIO


def run(opt, cfg):
    weight_file = Path(cfg.config['weight'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ISCNet_ScanNet(cfg, mode='test')
    dataloader = DataLoader(dataset=dataset,
                            num_workers=cfg.config['device']['num_workers'],
                            batch_size=1,
                            shuffle=False,
                            collate_fn=collate_fn,
                            worker_init_fn=my_worker_init_fn)

    network = get_model(cfg.config, device=device, dataset=dataset)
    checkpoint_io = CheckpointIO(os.fspath(weight_file.parent), model=network)

    try:
        load_dict = checkpoint_io.load(weight_file.name)
    except FileNotFoundError:
        load_dict = dict()
    epoch_it = load_dict.get('epoch_it', -1)
    it = load_dict.get('it', -1)

    network.cuda()
    network.eval()

    generator = Generator3D(network, threshold=0.5, resolution0=opt.retrieval_res / 4,
                            points_batch_size=opt.batch_points)

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

            voxels, ins_pc, overscan = voxels_from_scannet(ins_pc, c.box_centers[idx].cuda(), c.box_sizes[idx].cuda(),
                                                           c.axis_rectified[idx].cuda())

            vox_to_mesh(voxels[0].cpu().numpy(), out_scan_dir / f"{idx}_{c.shapenet_ids[idx]}_input")

            features = network.infer_c(ins_pc.transpose(1, 2))
            meshes = generator.generate_mesh(features, cls_codes=None, voxel_grid=voxels)[0]
            meshes = simplify_mesh(meshes)

            # output_pcd = output2[0, :, :3].detach().cpu().numpy()
            output_pcd_fn = tri_to_mesh(meshes, out_scan_dir / f"{idx}_{c.shapenet_ids[idx]}_output")

            ply_reader = vtk.vtkPLYReader()
            ply_reader.SetFileName(os.fspath(output_pcd_fn))
            ply_reader.Update()
            # get points from object
            polydata = ply_reader.GetOutput().GetPoints()
            # read points using vtk_to_numpy
            obj_points = torch.from_numpy(vtk_to_numpy(polydata.GetData()))
            obj_points = torch.matmul(obj_points * overscan.cpu(), transform_shapenet.T) * c.box_sizes[idx]
            obj_points = torch.matmul(obj_points, c.axis_rectified[idx]) + c.box_centers[idx]

            polydata.SetData(numpy_to_vtk(obj_points.numpy(), deep=True))
            instance_models.append(ply_reader)

            center_list.append(c.box_centers[idx].numpy())

            vectors = torch.diag(c.box_sizes[idx] / 2.) @ c.axis_rectified[idx]
            vector_list.append(vectors.numpy())

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
