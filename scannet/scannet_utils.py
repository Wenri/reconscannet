# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Ref: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts """
import csv
import os
import sys

import numpy as np

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    print("Please install the module 'plyfile' for PLY i/o, e.g.")
    print("pip install plyfile")
    sys.exit(-1)


class ShapeNetCat:
    chair_cat = {
        '03001627', '03632729', '20000027', '20000016', '03002210', '04331277', '03376595', '02738535',
        '04576002', '20000025', '20000026', '20000024', '20000020', '20000021', '20000023', '03260849',
        '20000018', '20000022', '20000019', '20000015', '03649674', '03002711', '04373704', '04099969'}
    table_cat = {
        '04381587', '20000037', '03090000', '03063968', '03238586', '04379243', '04603729', '20000038',
        '03092883', '20000041', '04301000', '03620967', '02874214', '03116530', '03179701', '03850492',
        '02699629', '02964075', '03246933', '02964196', '20000040', '04398951', '02894337', '03982430',
        '20000036', '03904060', '20000039'}
    cabinet_cat = {
        '20000009', '03237340', '03742115', '20000012', '03018349', '20000011', '20000008', '20000010',
        '20000013', '02933112'}
    display_cat = {
        '03211117', '03211616', '04152593', '03361380', '03196598', '03782190'}


def represents_int(s):
    """ if string s represents an int. """
    try:
        int(s)
        return True
    except ValueError:
        return False


def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = row[label_to]
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k): v for k, v in mapping.items()}
    return mapping


def read_mesh_vertices(filename):
    """ read XYZ for each vertex.
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:, 0] = plydata['vertex'].data['x']
        vertices[:, 1] = plydata['vertex'].data['y']
        vertices[:, 2] = plydata['vertex'].data['z']
    return vertices


def read_mesh_vertices_rgb(filename):
    """ read XYZ RGB for each vertex.
    Note: RGB values are in 0-255
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = plydata['vertex'].data['x']
        vertices[:, 1] = plydata['vertex'].data['y']
        vertices[:, 2] = plydata['vertex'].data['z']
        vertices[:, 3] = plydata['vertex'].data['red']
        vertices[:, 4] = plydata['vertex'].data['green']
        vertices[:, 5] = plydata['vertex'].data['blue']
    return vertices
