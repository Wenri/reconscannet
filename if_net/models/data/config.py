from torch import nn
from torchvision import transforms

from scannet.scannet_utils import chair_cat, table_cat
from .core import Shapes3dDataset
from .fields import IndexField, PointsField, PointCloudField, CategoryField, ImagesField, VoxelsField, \
    PartialPointCloudField, PartialJesseField
from .transforms import SubsamplePointcloud, PointcloudNoise, SubsamplePoints, SubselectPointcloud
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
    with_transforms = cfg['model']['use_camera']
    partial_transform = transforms.Compose([
        SubselectPointcloud(cfg['data']['pointcloud_n']),
        PointcloudNoise(cfg['data']['pointcloud_noise'])
    ])
    fields = {
        'points': PointsField(cfg['data']['points_file'], points_transform, with_transforms=with_transforms,
                              unpackbits=cfg['data']['points_unpackbits']),
        'pc': PointCloudField('pointcloud.npz', with_transforms=with_transforms),
        'partial': PartialPointCloudField('model', partial_transform, with_transforms=with_transforms),
        'partial_aug': PartialJesseField('model', partial_transform, with_transforms=with_transforms)
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
    cat_set = None if return_category else chair_cat
    # cat_set = table_cat

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
        # Input fields
        inputs_field = get_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = IndexField()

        if return_category:
            fields['category'] = CategoryField()

        dataset = Shapes3dDataset(
            dataset_folder, fields,
            split=split,
            categories=categories,
            cat_set=cat_set
        )
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])

    return dataset


def get_inputs_field(mode, cfg):
    """ Returns the inputs fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): config dictionary
    """
    input_type = cfg['data']['input_type']
    with_transforms = cfg['data']['with_transforms']

    if input_type is None:
        inputs_field = None
    elif input_type == 'img':
        if mode == 'train' and cfg['data']['img_augment']:
            resize_op = transforms.RandomResizedCrop(
                cfg['data']['img_size'], (0.75, 1.), (1., 1.))
        else:
            resize_op = transforms.Resize((cfg['data']['img_size']))

        transform = transforms.Compose([
            resize_op, transforms.ToTensor(),
        ])

        with_camera = cfg['data']['img_with_camera']

        if mode == 'train':
            random_view = True
        else:
            random_view = False

        inputs_field = ImagesField(
            cfg['data']['img_folder'], transform,
            with_camera=with_camera, random_view=random_view
        )
    elif input_type == 'pointcloud':
        transform = transforms.Compose([
            SubsamplePointcloud(cfg['data']['pointcloud_n']),
            PointcloudNoise(cfg['data']['pointcloud_noise'])
        ])
        with_transforms = cfg['data']['with_transforms']
        inputs_field = PointCloudField(
            cfg['data']['pointcloud_file'], transform,
            with_transforms=with_transforms
        )
    elif input_type == 'voxels':
        inputs_field = VoxelsField(
            cfg['data']['voxels_file']
        )
    elif input_type == 'idx':
        inputs_field = IndexField()
    else:
        raise ValueError(
            'Invalid input type (%s)' % input_type)
    return inputs_field


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
