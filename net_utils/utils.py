# Utility functions during training and testing.
# author: ynie
# date: Feb, 2020

import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def initiate_environment(config):
    """
    initiate randomness.
    :param config:
    :return:
    """
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def load_device(cfg):
    """
    load device settings
    :param config:
    :return:
    """
    if cfg.config['device']['use_gpu'] and torch.cuda.is_available():
        cfg.log_string('GPU mode is on.')
        cfg.log_string('GPU Ids: %s used.' % (cfg.config['device']['gpu_ids']))
        return torch.device("cuda")
    else:
        cfg.log_string('CPU mode is on.')
        return torch.device("cpu")


class AverageMeter(object):
    """
    Computes ans stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val  # current value
        if not isinstance(val, list):
            self.sum += val * n  # accumulated sum, n = batch_size
            self.count += n  # accumulated count
        else:
            self.sum += sum(val)
            self.count += len(val)
        self.avg = self.sum / self.count  # current average value


class LossRecorder(object):
    def __init__(self, batch_size=1):
        """
        Log loss data
        :param config: configuration file.
        :param phase: train, validation or test.
        """
        self._batch_size = batch_size
        self._loss_recorder = {}

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def loss_recorder(self):
        return self._loss_recorder

    def update_loss(self, loss_dict):
        for key, item in loss_dict.items():
            if key not in self._loss_recorder:
                self._loss_recorder[key] = AverageMeter()
            self._loss_recorder[key].update(item, self._batch_size)


class LogBoard(object):
    def __init__(self):
        self.writer = SummaryWriter()
        self.iter = 1

    def update(self, value_dict, step_len, phase):
        n_iter = self.iter * step_len
        for key, item in value_dict.items():
            self.writer.add_scalar(key + '/' + phase, item, n_iter)
        self.iter += 1
