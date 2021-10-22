import glob
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

from external import binvox_rw
from net_utils.voxel_util import roty
from .core import Field


class IndexField(Field):
    """ Basic index field."""

    def load(self, model_path, idx, category):
        """ Loads the index field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        """
        return idx

    def check_complete(self, files):
        """ Check if field is complete.

        Args:
            files: files
        """
        return True


class CategoryField(Field):
    """ Basic category field."""

    def load(self, model_path, idx, category):
        """ Loads the category field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        """
        return category

    def check_complete(self, files):
        """ Check if field is complete.

        Args:
            files: files
        """
        return True


class ImagesField(Field):
    """ Image Field.

    It is the field used for loading images.

    Args:
        folder_name (str): folder name
        transform (list): list of transformations applied to loaded images
        extension (str): image extension
        random_view (bool): whether a random view should be used
        with_camera (bool): whether camera data should be provided
    """

    def __init__(self, folder_name, transform=None,
                 extension='jpg', random_view=True, with_camera=False):
        self.folder_name = folder_name
        self.transform = transform
        self.extension = extension
        self.random_view = random_view
        self.with_camera = with_camera

    def load(self, model_path, idx, category):
        """ Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        """
        folder = os.path.join(model_path, self.folder_name)
        files = glob.glob(os.path.join(folder, '*.%s' % self.extension))
        files.sort()

        if self.random_view:
            idx_img = random.randint(0, len(files) - 1)
        else:
            idx_img = 0
        filename = files[idx_img]

        image = Image.open(filename).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        data = {
            None: image
        }

        if self.with_camera:
            camera_file = os.path.join(folder, 'cameras.npz')
            camera_dict = np.load(camera_file)
            Rt = camera_dict['world_mat_%d' % idx_img].astype(np.float32)
            K = camera_dict['camera_mat_%d' % idx_img].astype(np.float32)
            data['world_mat'] = Rt
            data['camera_mat'] = K

        return data

    def check_complete(self, files):
        """ Check if field is complete.

        Args:
            files: files
        """
        complete = (self.folder_name in files)
        # TODO: check camera
        return complete


# 3D Fields
class PointsField(Field):
    """ Point Field.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape.

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the
            points tensor
        with_transforms (bool): whether scaling and rotation data should be
            provided

    """

    def __init__(self, file_name, transform=None, with_transforms=False, unpackbits=False):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms
        self.unpackbits = unpackbits

    def load(self, model_path, idx, category):
        """ Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        """
        file_path = os.path.join(model_path, self.file_name)

        points_dict = np.load(file_path)
        points = points_dict['points']
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)
        else:
            points = points.astype(np.float32)

        occupancies = points_dict['occupancies']
        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)

        data = {
            None: points,
            'occ': occupancies,
        }

        if self.with_transforms:
            data['loc'] = points_dict['loc'].astype(np.float32)
            data['scale'] = points_dict['scale'].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return data


class VoxelsField(Field):
    """ Voxel field class.

    It provides the class used for voxel-based data.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
    """

    def __init__(self, file_name, transform=None):
        self.file_name = file_name
        self.transform = transform

    def load(self, model_path, idx, category):
        """ Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        """
        file_path = os.path.join(model_path, self.file_name)

        with open(file_path, 'rb') as f:
            voxels = binvox_rw.read_as_3d_array(f)
        voxels = voxels.data.astype(np.float32)

        if self.transform is not None:
            voxels = self.transform(voxels)

        return voxels

    def check_complete(self, files):
        """ Check if field is complete.

        Args:
            files: files
        """
        complete = (self.file_name in files)
        return complete


class PointCloudField(Field):
    """ Point cloud field.

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        with_transforms (bool): whether scaling and rotation dat should be
            provided
    """

    def __init__(self, file_name, transform=None, with_transforms=False):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)

        data = {
            None: points,
            'normals': normals,
        }

        if self.with_transforms:
            data['loc'] = pointcloud_dict['loc'].astype(np.float32)
            data['scale'] = pointcloud_dict['scale'].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        """ Check if field is complete.

        Args:
            files: files
        """
        complete = (self.file_name in files)
        return complete


class PartialPointCloudField(Field):
    """ Partial Point cloud field.

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        with_transforms (bool): whether scaling and rotation dat should be
            provided
    """

    def __init__(self, file_name, transform=None, aug_ratio=0.05, is_training=True):
        self.file_name = file_name
        self.transform = transform
        self.aug_ratio = aug_ratio
        self.is_training = is_training
        self._rand = random.Random()

    def load(self, model_path, idx, category):
        """ Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        """

        data = None

        if not self.is_training or self._rand.random() < self.aug_ratio:
            data = self.load_jesse(model_path)

        if data is None:
            data = self.load_gbc(model_path)

        if self.transform is not None:
            data = self.transform(data)

        return data

    def load_gbc(self, model_path):
        file_path = os.path.join(model_path, self.file_name)
        if self.is_training:
            pointcloud_file = self._rand.choice(glob.glob(os.path.join(file_path, 'seed_[0-9A-Z]-XYZ.npy')))
        else:
            pointcloud_file = os.path.join(file_path, 'render-XYZ.npy')

        return {
            None: np.load(pointcloud_file, mmap_mode='r')[:, :3].astype(np.float32),
            'aug': np.zeros(shape=(), dtype=np.bool_)
        }

    def load_jesse(self, model_path):
        model_path = Path(model_path)
        model_category = model_path.parent
        file_path = self.search_model_filename(Path(*model_category.parent.parts[:-1], 'jesse'),
                                               category=model_category.name, model=model_path.name)

        if not file_path:
            return None

        pointcloud_file = trimesh.load(file_path)

        total_pc = torch.from_numpy(pointcloud_file.vertices)
        total_pc = total_pc @ roty.T.to(dtype=total_pc.dtype)

        return {
            None: total_pc.float().numpy(),
            'aug': np.ones(shape=(), dtype=np.bool_)
        }

    def load_shapenet(self, category, model):
        shapenet_path = Path('ShapeNetCore.v2', category, model, 'models', 'model_normalized.obj')
        shapenet_file = trimesh.load(shapenet_path, force='mesh')
        bb_max, bb_min = np.max(shapenet_file.vertices, axis=0), np.min(shapenet_file.vertices, axis=0)
        total_size = (bb_max - bb_min).max()
        centers = (bb_min + bb_max) / 2
        with shapenet_path.with_suffix('.json').open('r') as f:
            shapenet_json = json.load(f)
        return shapenet_file, shapenet_json

    @staticmethod
    def search_model_filename(model_path, category, model):
        file_path = Path(model_path, category, model + '.obj')
        if not file_path.is_file():
            if not file_path.with_suffix('').is_dir():
                category = '*'
            file_path = glob.glob(f'{os.fspath(model_path)}/{category}/{model}/*.obj')
            if file_path:
                file_path, = file_path
        return file_path

    def check_complete(self, files):
        """ Check if field is complete.

        Args:
            files: files
        """
        complete = (self.file_name in files)
        return complete


# NOTE: this will produce variable length output.
# You need to specify collate_fn to make it work with a data laoder
class MeshField(Field):
    """ Mesh field.

    It provides the field used for mesh data. Note that, depending on the
    dataset, it produces variable length output, so that you need to specify
    collate_fn to make it work with a data loader.

    Args:
        file_name (str): file name
        transform (list): list of transforms applied to data points
    """

    def __init__(self, file_name, transform=None):
        self.file_name = file_name
        self.transform = transform

    def load(self, model_path, idx, category):
        """ Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        """
        file_path = os.path.join(model_path, self.file_name)

        mesh = trimesh.load(file_path, process=False)
        if self.transform is not None:
            mesh = self.transform(mesh)

        data = {
            'verts': mesh.vertices,
            'faces': mesh.faces,
        }

        return data

    def check_complete(self, files):
        """ Check if field is complete.

        Args:
            files: files
        """
        complete = (self.file_name in files)
        return complete
