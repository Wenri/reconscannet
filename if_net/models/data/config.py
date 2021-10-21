from torch import nn
from torchvision import transforms

from scannet.scannet_utils import ShapeNetCat
from .core import Shapes3dDataset
from .fields import IndexField, PointsField, CategoryField, VoxelsField, \
    PartialPointCloudField
from .transforms import PointcloudNoise, SubsamplePoints, SubselectPointcloud
from ..if_net import IFNet


def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']
    encoder_latent = cfg['model']['encoder_latent']
    dim = cfg['data']['dim']
    z_dim = cfg['model']['z_dim']
    c_dim = cfg['model']['c_dim']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    encoder_latent_kwargs = cfg['model']['encoder_latent_kwargs']

    model = IFNet(cfg)

    return model


def get_data_fields(mode, cfg):
    """ Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    """
    points_transform = SubsamplePoints(cfg['data']['points_subsample'])
    with_transforms = cfg['data']['with_transforms']
    is_training = not mode == 'test'
    partial_transform = transforms.Compose([
        SubselectPointcloud(cfg['data']['pointcloud_n']),
        PointcloudNoise(cfg['data']['pointcloud_noise'])
    ])
    fields = {
        'points': PointsField(cfg['data']['points_file'], points_transform, with_transforms=with_transforms,
                              unpackbits=cfg['data']['points_unpackbits']),
        'partial': PartialPointCloudField('model', partial_transform, with_transforms=with_transforms,
                                          is_training=is_training),
        # 'pc': PointCloudField('pointcloud.npz', with_transforms=with_transforms),
    }

    voxels_file = cfg['data'].get('voxels_file')
    if voxels_file:
        fields['voxels'] = VoxelsField(voxels_file)

    if mode in {'val', 'test'}:
        points_iou_file = cfg['data']['points_iou_file']
        if points_iou_file is not None:
            fields['points_iou'] = PointsField(
                points_iou_file,
                with_transforms=with_transforms,
                unpackbits=cfg['data']['points_unpackbits'],
            )

    return fields


# Datasets
def get_dataset(mode, cfg, return_idx=False):
    """ Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    """
    # method = cfg['method']
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    categories = cfg['data']['classes']
    return_category = cfg['data']['use_cls_for_completion']
    cat_set = getattr(ShapeNetCat, categories) if categories else None
    is_training = not mode == 'test'

    # Get split
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
    }

    split = splits[mode]

    # Create dataset
    if dataset_type == 'Shapes3D':
        # Dataset fields
        # Method specific fields (usually correspond to output)
        fields = get_data_fields(mode, cfg)

        if return_idx:
            fields['idx'] = IndexField()

        if return_category:
            fields['category'] = CategoryField()

        dataset = Shapes3dDataset(
            dataset_folder, fields,
            split=split,
            cat_set=cat_set,
            is_training=is_training
        )
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])

    return dataset


def get_preprocessor(cfg, dataset=None, device=None):
    """ Returns preprocessor instance.

    Args:
        cfg (dict): config dictionary
        dataset (dataset): dataset
        device (device): pytorch device
    """
    p_type = cfg['preprocessor']['type']
    # cfg_path = cfg['preprocessor']['config']
    # model_file = cfg['preprocessor']['model_file']

    if p_type is None:
        preprocessor = None
    else:
        raise ValueError('Invalid Preprocessor %s' % p_type)

    return preprocessor
