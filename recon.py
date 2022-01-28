import argparse
import os
import sys
from pathlib import Path, PurePath
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
from if_net.models.data.config import get_model
from if_net.models.data.core import list_categories
from if_net.models.generator import Generator3D
from net_utils.utils import initiate_environment
from net_utils.voxel_util import voxels_from_scannet, transform_shapenet
from scannet.scannet_utils import ShapeNetCat
from scannet.visualization.vis_for_demo import Vis_base
from utils.checkpoints import CheckpointIO


def run(opt, cfg):
    export_db = {}
    with open('name.txt', 'r') as f:
        for p in f:
            pf = PurePath(p.strip())
            if pf.name != 'points.npz':
                continue
            s = pf.parent.name
            if not s.startswith('scan'):
                continue
            scanid, objid, clsid = s.split('_')
            curid = export_db.setdefault((int(scanid[4:]), int(objid)), clsid)
            assert curid == clsid

    weight_file = Path(cfg.config['weight'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    categories = {c: idx for idx, c in enumerate(list_categories(cfg.config['data']['path']))}

    dataset = ISCNet_ScanNet(cfg, mode='test', split='train')
    dataloader = DataLoader(dataset=dataset,
                            num_workers=cfg.config['device']['num_workers'],
                            batch_size=1,
                            shuffle=False,
                            collate_fn=collate_fn,
                            worker_init_fn=my_worker_init_fn)

    network = get_model(cfg.config, device=device, dataset=dataset)
    checkpoint_io = CheckpointIO(os.fspath(weight_file.parent), model=network)
    cat_set = cfg.config['data']['classes']
    cat_set = getattr(ShapeNetCat, cat_set) if cat_set else None
    # cat_set = ShapeNetCat.cabinet_cat | ShapeNetCat.table_cat | ShapeNetCat.chair_cat

    try:
        checkpoint_io.load(weight_file.name)
    except FileNotFoundError:
        print('Checkpoint File Not Found Error', file=sys.stderr)
        return -1

    network.cuda()
    network.eval()

    generator = Generator3D(network, points_batch_size=opt.batch_points, threshold=cfg.config['test']['threshold'],
                            resolution0=cfg.config['generation']['resolution_0'],
                            simplify_nfaces=cfg.config['generation']['simplify_nfaces'],
                            refinement_step=cfg.config['generation']['refinement_step'],
                            padding=0)

    for cur_iter, data in enumerate(dataloader):
        # if cur_iter <= 200:
        #     continue

        bid = 0
        c = SimpleNamespace(**{k: v[bid] for k, v in get_bbox(cfg.dataset_config, **data).items()})

        instance_models = []
        center_list = []
        vector_list = []

        out_scan_dir = opt.output_dir / f'scan_{c.scan_idx}'
        print(f'scan_{c.scan_idx}')

        for idx in c.box_label_mask.nonzero(as_tuple=True)[0]:
            # if cat_set is not None and c.shapenet_catids[idx] not in cat_set:
            #     continue

            ins_id = c.object_instance_labels[idx]
            ins_pc = c.point_clouds[c.point_instance_labels == ins_id].cuda()

            clsid = export_db.get((c.scan_idx.item(), idx.item()))
            if clsid is None:
                continue

            assert c.shapenet_ids[idx].startswith(clsid)

            out_scan_dir.mkdir(exist_ok=True)

            voxels, ins_pc, overscan = voxels_from_scannet(ins_pc, c.box_centers[idx].cuda(), c.box_sizes[idx].cuda(),
                                                           c.axis_rectified[idx].cuda())

            vox_to_mesh(voxels[0].cpu().numpy(), out_scan_dir / f"{idx}_{c.shapenet_ids[idx]}_input")

            cat_idx = torch.as_tensor(categories[c.shapenet_catids[idx]])
            features = network.infer_c(ins_pc.transpose(1, 2), cls_codes_for_completion=cat_idx.unsqueeze(0))
            meshes = generator.generate_mesh(features, voxel_grid=voxels)[0]

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

    return 0


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
    parser.add_argument('--batch_points', default=100000, type=int)
    return parser.parse_args()


def main(args):
    cfg = CONFIG(args.config, dataset_config=ScannetConfig())

    initiate_environment(cfg.config)

    '''Configuration'''
    cfg.log_string('Loading configurations.')
    cfg.log_string(cfg.config)
    cfg.write_config()

    return run(args, cfg)


if __name__ == '__main__':
    sys.exit(main(parse_args()))
