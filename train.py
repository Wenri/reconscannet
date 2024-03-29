import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from adabelief_pytorch import AdaBelief
from tqdm import tqdm

from configs.config_utils import CONFIG
from if_net.models.data.config import get_dataset, get_model
from if_net.models.data.core import collate_remove_none, worker_init_fn
from if_net.models.training import Trainer
from utils.checkpoints import CheckpointIO


def evaluate(trainer, val_loader):
    torch.cuda.empty_cache()
    metric_val = []
    total_invalid = 0
    total_fixed = 0
    teval = tqdm(val_loader, unit='images', unit_scale=val_loader.batch_size, file=sys.stdout, dynamic_ncols=True)
    for batch in teval:
        val, invalid_id, fixed_id = trainer.eval_step(batch)
        metric_val += [v for idx, v in enumerate(val) if idx not in invalid_id]
        total_invalid += len(invalid_id)
        total_fixed += len(fixed_id)
        teval.set_description(
            '[metric val: %.4f, FIX: %d, INV: %d]' % (sum(metric_val) / len(metric_val), total_fixed, total_invalid))
    metric_val = sum(metric_val) / len(metric_val)
    print('total metric val:  %.4f' % metric_val)
    return metric_val


def train_one_epoch(train_loader, trainer, epoch_it, print_every):
    torch.cuda.empty_cache()
    total_aug = 0
    fix_number = 0
    inv_stat = []

    for it, batch in enumerate(train_loader):
        loss, aug, fixed_id, invalid_id = trainer.train_step(batch)
        fix_number += len(fixed_id)
        inv_stat.extend(invalid_id.values())
        total_aug += aug
        # logger.add_scalar('train/loss', loss, it)
        # Print output
        if print_every > 0 and (it % print_every) == 0:
            print('[Epoch %02d] it=%03d, lr %.6f, loss=%.4f, aug_ratio=%.2f, FIX=%d, INV=%d:'
                  % (epoch_it, it, trainer.get_lr(), loss, total_aug / print_every / train_loader.batch_size,
                     fix_number, len(inv_stat)), end=' ')
            inv_stat.sort()
            for reason in inv_stat:
                print('%.2f' % reason, end=' ')
            print()
            total_aug = 0
            fix_number = 0
            inv_stat.clear()


def main(args):
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")
    cfg = CONFIG(args.config).config

    # Set t0
    t0 = time.time()

    # Shorthands
    out_dir = cfg['training']['out_dir']
    batch_size = cfg['training']['batch_size']
    exit_after = args.exit_after

    # Output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Dataset
    train_dataset = get_dataset('train', cfg)
    val_dataset = get_dataset('val', cfg)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=16, shuffle=True, prefetch_factor=2,
        collate_fn=collate_remove_none, worker_init_fn=worker_init_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=4, num_workers=4, shuffle=False,
        collate_fn=collate_remove_none, worker_init_fn=worker_init_fn)

    # For visualizations
    # vis_loader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=12, shuffle=True,
    #     collate_fn=collate_remove_none,
    #     worker_init_fn=worker_init_fn)
    # data_vis = next(iter(vis_loader))

    # Model
    model = get_model(cfg, device=device, dataset=train_dataset)

    # Intialize training
    # optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer = AdaBelief(model.parameters(), lr=args.lr)
    trainer = Trainer(model, exp_name='if_net_scannet', optimizer=optimizer, device=device,
                      warmup_iters=100, balance_weight=cfg['training']['balance_weight'])

    checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
    try:
        load_dict = checkpoint_io.load('model.pt')
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print('checkpoint_io load err: ', e, file=sys.stderr)
        load_dict = dict()
    epoch_it = load_dict.get('epoch_it', -1)
    metric_val_best = load_dict.get('loss_val_best', -np.inf)
    # metric_val_best = -np.inf

    print('Current best validation metric: %.8f' % metric_val_best)

    # TODO: reintroduce or remove scheduler?
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4000,
    #                                       gamma=0.1, last_epoch=epoch_it)
    # logger = SummaryWriter(os.path.join(out_dir, 'logs'))

    # Shorthands
    print_every = cfg['training']['print_every']
    backup_every = cfg['training']['backup_every']
    validate_every = cfg['training']['validate_every']
    visualize_every = cfg['training']['visualize_every']

    # Print model
    nparameters = sum(p.numel() for p in model.parameters())
    print(model)
    print('Total number of parameters: %d' % nparameters)

    print('Total dataset length: %d, valid length: %d' % (len(train_dataset), len(train_dataset._valid_map)))

    # trainer.set_lr(1e-4)
    while True:
        epoch_it += 1
        train_one_epoch(train_loader, trainer, epoch_it, print_every)
        trainer.lr_scheduler = None

        # Backup if necessary
        if backup_every > 0 and (epoch_it % backup_every) == 0:
            metric_val = evaluate(trainer, val_loader)
            if not np.isfinite(metric_val_best) or metric_val > metric_val_best:
                metric_val_best = metric_val
                backup_name = 'model_%d_%.4f+.pt' % (epoch_it, metric_val)
                checkpoint_io.sym_link(backup_name)
            else:
                backup_name = 'model_%d_%.4f.pt' % (epoch_it, metric_val)
            print('Backup checkpoint')
            checkpoint_io.save(backup_name, epoch_it=epoch_it, loss_val_best=metric_val_best)

        # Save checkpoint
        print('Saving checkpoint')
        checkpoint_io.save('model.pt', epoch_it=epoch_it, loss_val_best=metric_val_best)


def parse_args():
    """PARAMETERS"""
    base_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(
        description='Train a 3D reconstruction model.'
    )
    parser.add_argument('--config', type=Path, default=base_dir / 'configs' / 'config_files' / 'if_net.yaml',
                        help='Path to config file.')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    parser.add_argument('--exit-after', type=int, default=-1,
                        help='Checkpoint and exit after specified number of seconds'
                             'with exit code 2.')
    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')

    return parser.parse_args()


if __name__ == '__main__':
    matplotlib.use('Agg')
    sys.exit(main(parse_args()))
