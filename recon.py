import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import vtkmodules.all as vtk
from torch.utils.data import DataLoader
from vtkmodules.util.numpy_support import numpy_to_vtk

from configs.config_utils import CONFIG
from configs.scannet_config import ScannetConfig
from dataloader import ISCNet_ScanNet, collate_fn, my_worker_init_fn
from msn.model import MSN
from msn.utils import weights_init
from net_utils.box_util import get_3d_box_cuda
from net_utils.libs import flip_axis_to_camera_cuda, flip_axis_to_depth_cuda
from net_utils.utils import initiate_environment
from scannet.visualization.vis_for_demo import Vis_base


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

    transform_shapenet = torch.tensor([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])

    for cur_iter, data in enumerate(dataloader):
        bid = 0
        c = SimpleNamespace(**{k: v[bid] for k, v in get_bbox(cfg.dataset_config, **data).items()})

        instance_models = []
        center_list = []
        vector_list = []

        for idx in c.box_label_mask.nonzero(as_tuple=True)[0]:
            ins_id = c.object_instance_labels[idx]
            ins_pc = c.point_clouds[c.point_instance_labels == ins_id]

            point_clouds = torch.matmul(ins_pc - c.box_centers[idx], c.axis_rectified[idx].T)
            point_clouds = torch.matmul(point_clouds / c.box_sizes[idx], transform_shapenet)
            print(torch.min(point_clouds), torch.max(point_clouds))

            point_clouds = point_clouds.T.cuda()

            output1, output2, expansion_penalty = network(point_clouds.unsqueeze(0))

            points_array = numpy_to_vtk(output2[0, :, :3].detach().cpu().numpy(), deep=True)
            polydata = vtk.vtkPolyData()
            vtkpoints = vtk.vtkPoints()
            vtkpoints.SetData(points_array)
            polydata.SetPoints(vtkpoints)
            instance_models.append(polydata)
            center_list.append(c.box_centers[idx].numpy())

            vectors = torch.diag(c.box_sizes[idx] / 2.) @ c.axis_rectified[idx]
            vector_list.append(vectors.numpy())

            print(output2.shape)

        scene = Vis_base(scene_points=c.point_clouds, instance_models=instance_models, center_list=center_list,
                         vector_list=vector_list)
        camera_center = np.array([0, -3, 3])
        scene.visualize(centroid=camera_center, offline=False, save_path=Path('out', 'pred.png'))


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


def main():
    args = parse_args()
    cfg = CONFIG(args.config, dataset_config=ScannetConfig())

    initiate_environment(cfg.config)

    '''Configuration'''
    cfg.log_string('Loading configurations.')
    cfg.log_string(cfg.config)
    cfg.write_config()

    run(args, cfg)


if __name__ == '__main__':
    main()
