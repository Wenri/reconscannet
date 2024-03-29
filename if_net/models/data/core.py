import logging
import os
import random
from pathlib import Path

import numpy as np
import yaml
from torch.utils import data

logger = logging.getLogger(__name__)


# Fields
class Field(object):
    """ Data fields class.
    """

    def load(self, data_path, idx, category):
        """ Loads a data point.

        Args:
            data_path (str): path to data file
            idx (int): index of data point
            category (int): index of category
        """
        raise NotImplementedError

    def check_complete(self, files):
        """ Checks if set is complete.

        Args:
            files: files
        """
        raise NotImplementedError


class Shapes3dDataset(data.Dataset):
    """ 3D Shapes dataset class.
    """

    def __init__(self, dataset_folder, fields, split=None, cat_set=None, no_except=False, transform=None,
                 is_training=True):
        """ Initialization of the the 3D shape dataset.

        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            split (str): which split is used
            categories (list): list of categories to use
            no_except (bool): no exception
            transform (callable): transformation applied to data points
        """
        # Attributes
        self.dataset_folder = dataset_folder
        self.fields = fields
        self.no_except = no_except
        self.transform = transform
        self._rand = random.Random()
        self.cat_set = cat_set
        self.is_training = is_training

        categories = list_categories(dataset_folder)
        # Read metadata file
        metadata_file = os.path.join(dataset_folder, 'metadata.yaml')

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = yaml.load(f)
        else:
            self.metadata = {
                c: {'id': c, 'name': 'n/a'} for c in categories
            }

            # Set index
        for c_idx, c in enumerate(categories):
            self.metadata[c]['idx'] = c_idx

        # Get all models
        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(dataset_folder, c)
            if not os.path.isdir(subpath):
                logger.warning('Category %s does not exist in dataset.' % c)

            split_file = os.path.join(subpath, split + '.lst')
            with open(split_file, 'r') as f:
                models_c = f.read().split('\n')

            self.models += [{'category': c, 'model': m} for m in models_c]

        self._valid_map = list(self.update_valid())

    def update_valid(self):
        for i, m in enumerate(self.models):
            fname = 'seed_0-XYZ.npy' if self.is_training else 'render-XYZ.npy'
            subpath = Path(self.dataset_folder, m['category'], m['model'], 'model', fname)
            if subpath.exists() and (self.cat_set is None or m['category'] in self.cat_set):
                yield i

    def __len__(self):
        """ Returns the length of the dataset.
        """
        self._valid_map = list(self.update_valid())
        if self.is_training:
            self._rand.shuffle(self._valid_map)
        return len(self.models) if self.is_training else len(self._valid_map)

    def __getitem__(self, idx):
        """ Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        """
        idx = self._valid_map[idx % len(self._valid_map)]
        category = self.models[idx]['category']
        model = self.models[idx]['model']
        c_idx = self.metadata[category]['idx']

        model_path = os.path.join(self.dataset_folder, category, model)
        data = {}

        for field_name, field in self.fields.items():
            try:
                field_data = field.load(model_path, idx, c_idx)
            except Exception as ex:
                if self.no_except:
                    logger.warn(
                        'Error occured when loading field %s of model %s: %s'
                        % (field_name, model, str(ex))
                    )
                    return None
                else:
                    raise

            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data[field_name] = v
                    else:
                        data['%s.%s' % (field_name, k)] = v
            else:
                data[field_name] = field_data

        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_model_dict(self, idx):
        return self.models[idx]

    def test_model_complete(self, category, model):
        """ Tests if model is complete.

        Args:
            model (str): modelname
        """
        model_path = os.path.join(self.dataset_folder, category, model)
        files = os.listdir(model_path)
        for field_name, field in self.fields.items():
            if not field.check_complete(files):
                logger.warn('Field "%s" is incomplete: %s'
                            % (field_name, model_path))
                return False

        return True


def collate_remove_none(batch):
    """ Collater that puts each data field into a tensor with outer dimension
        batch size.

    Args:
        batch: batch
    """

    batch = list(filter(lambda x: x is not None, batch))
    return data.dataloader.default_collate(batch)


def worker_init_fn(worker_id):
    """ Worker init function to ensure true randomness.
    """
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)


def list_categories(dataset_folder):
    categories = os.listdir(dataset_folder)
    categories = [c for c in categories
                  if os.path.isdir(os.path.join(dataset_folder, c))]
    categories.sort()
    return categories
