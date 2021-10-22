from itertools import chain, product

import numpy as np
from scipy.spatial.transform import Rotation as R


def generate_rotmatrix():
    def m_xyz(deg):
        return tuple(R.from_euler(axis, deg, degrees=True).as_matrix() for axis in 'xyz')

    def chaindedup(mat_list):
        mats = set(np.asarray(a, dtype=np.int).tobytes('C') for a in mat_list)
        return tuple(np.reshape(np.frombuffer(buf, dtype=np.int), (3, 3), order='C') for buf in mats)

    base = chaindedup(chain.from_iterable(m_xyz(a) for a in (0, 90, 180, 270)))
    level2 = chaindedup(np.matmul(y, x) for x, y in product(base, repeat=2))
    level3 = chaindedup(np.matmul(y, x) for x, y in product(base, level2))

    return level3


# Transforms
class PointcloudNoise(object):
    """ Point cloud noise transformation class.

    It adds noise to point cloud data.

    Args:
        stddev (int): standard deviation
    """

    def __init__(self, stddev):
        self.stddev = stddev
        self.rand = np.random.default_rng()

    def __call__(self, data):
        """ Calls the transformation.

        Args:
            data (dictionary): data dictionary
        """
        data_out = data.copy()
        points = data[None]
        noise = self.rand.normal(size=points.shape, scale=self.stddev)
        data_out[None] = (points + noise).astype(np.float32)
        return data_out


class SubsamplePointcloud(object):
    """ Point cloud subsampling transformation class.

    It subsamples the point cloud data.

    Args:
        N (int): number of points to be subsampled
    """

    def __init__(self, N):
        self.N = N
        self.rand = np.random.default_rng()

    def __call__(self, data):
        """ Calls the transformation.

        Args:
            data (dict): data dictionary
        """
        data_out = data.copy()
        points = data[None]
        total = points.shape[0]

        indices = self.rand.permutation(total)
        if indices.shape[0] < self.N:
            indices = np.concatenate([indices, self.rand.integers(total, size=self.N - total)])
        indices = indices[:self.N]

        data_out[None] = points[indices, :]

        normals = data.get('normals')
        if normals is not None:
            data_out['normals'] = normals[indices, :]

        return data_out


class SubselectPointcloud(object):
    """ Point cloud subsampling transformation class.

    It subsamples the point cloud data.

    Args:
        N (int): number of points to be subsampled
    """

    def __init__(self, N):
        self.N = N
        self._rand = np.random.default_rng()

    def __call__(self, data):
        """ Calls the transformation.

        Args:
            data (dict): data dictionary
        """
        data_out = data.copy()
        points = data[None]
        total = points.shape[0]

        if total < self.N:
            indices = self._rand.permutation(total)
            indices = np.concatenate([indices, self._rand.integers(total, size=self.N - total)])
        else:
            indices = slice(self.N)

        data_out[None] = points[indices, :]

        normals = data.get('normals')
        if normals is not None:
            data_out['normals'] = normals[indices, :]

        return data_out


class SubsamplePoints(object):
    """ Points subsampling transformation class.

    It subsamples the points data.

    Args:
        N (int): number of points to be subsampled
    """

    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        """ Calls the transformation.

        Args:
            data (dictionary): data dictionary
        """
        points = data[None]
        occ = data['occ']

        data_out = data.copy()
        if isinstance(self.N, int):
            idx = np.random.randint(points.shape[0], size=self.N)
            data_out.update({
                None: points[idx, :],
                'occ': occ[idx],
            })
        else:
            Nt_out, Nt_in = self.N
            occ_binary = (occ >= 0.5)
            points0 = points[~occ_binary]
            points1 = points[occ_binary]

            idx0 = np.random.randint(points0.shape[0], size=Nt_out)
            idx1 = np.random.randint(points1.shape[0], size=Nt_in)

            points0 = points0[idx0, :]
            points1 = points1[idx1, :]
            points = np.concatenate([points0, points1], axis=0)

            occ0 = np.zeros(Nt_out, dtype=np.float32)
            occ1 = np.ones(Nt_in, dtype=np.float32)
            occ = np.concatenate([occ0, occ1], axis=0)

            volume = occ_binary.sum() / len(occ_binary)
            volume = volume.astype(np.float32)

            data_out.update({
                None: points,
                'occ': occ,
                'volume': volume,
            })
        return data_out
