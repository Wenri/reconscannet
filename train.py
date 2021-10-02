import argparse
import os
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import optim

from configs.config_utils import CONFIG
from if_net.models.data.config import get_dataset, get_model
from if_net.models.data.core import collate_remove_none, worker_init_fn
from if_net.models.training import Trainer
from utils.checkpoints import CheckpointIO


def main(args):
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")
    cfg = CONFIG(args.config).config

    # Set t0
    t0 = time.time()

    # Shorthands
    out_dir = cfg['training']['out_dir']
    batch_size = cfg['training']['batch_size']
    backup_every = cfg['training']['backup_every']
    exit_after = args.exit_after

    model_selection_metric = cfg['training']['model_selection_metric']
    if cfg['training']['model_selection_mode'] == 'maximize':
        model_selection_sign = 1
    elif cfg['training']['model_selection_mode'] == 'minimize':
        model_selection_sign = -1
    else:
        raise ValueError('model_selection_mode must be '
                         'either maximize or minimize.')

    # Output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Dataset
    train_dataset = get_dataset('train', cfg)
    val_dataset = get_dataset('val', cfg)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=16, shuffle=True,
        collate_fn=collate_remove_none,
        worker_init_fn=worker_init_fn, prefetch_factor=10)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=10, num_workers=8, shuffle=False,
        collate_fn=collate_remove_none,
        worker_init_fn=worker_init_fn)

    # For visualizations
    vis_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=12, shuffle=True,
        collate_fn=collate_remove_none,
        worker_init_fn=worker_init_fn)
    data_vis = next(iter(vis_loader))

    # Model
    model = get_model(cfg, device=device, dataset=train_dataset)

    # Intialize training
    # optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = AdaBelief(model.parameters(), lr=1e-4)
    trainer = Trainer(model, exp_name='if_net_scannet', optimizer=optimizer, device=device,
                      balance_weight=cfg['training']['balance_weight'])

    checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
    try:
        load_dict = checkpoint_io.load('model.pt')
    except FileNotFoundError:
        load_dict = dict()
    epoch_it = load_dict.get('epoch_it', -1)
    it = load_dict.get('it', -1)
    metric_val_best = load_dict.get(
        'loss_val_best', -model_selection_sign * np.inf)

    # Hack because of previous bug in code
    # TODO: remove, because shouldn't be necessary
    if metric_val_best == np.inf or metric_val_best == -np.inf:
        metric_val_best = -model_selection_sign * np.inf

    # TODO: remove this switch
    # metric_val_best = -model_selection_sign * np.inf

    print('Current best validation metric (%s): %.8f'
          % (model_selection_metric, metric_val_best))

    # TODO: reintroduce or remove scheduler?
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4000,
    #                                       gamma=0.1, last_epoch=epoch_it)
    # logger = SummaryWriter(os.path.join(out_dir, 'logs'))

    # Shorthands
    print_every = cfg['training']['print_every']
    checkpoint_every = cfg['training']['checkpoint_every']
    validate_every = cfg['training']['validate_every']
    visualize_every = cfg['training']['visualize_every']

    # Print model
    nparameters = sum(p.numel() for p in model.parameters())
    print(model)
    print('Total number of parameters: %d' % nparameters)

    print('Total dataset length: %d' % len(train_dataset))

    while True:
        epoch_it += 1
        #     scheduler.step()

        for batch in train_loader:
            it += 1
            loss = trainer.train_step(batch)
            # logger.add_scalar('train/loss', loss, it)

            # Print output
            if print_every > 0 and (it % print_every) == 0:
                print('[Epoch %02d] it=%03d, loss=%.4f'
                      % (epoch_it, it, loss))

        # Save checkpoint
        print('Saving checkpoint')
        checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it, loss_val_best=metric_val_best)

        # Backup if necessary
        if backup_every > 0 and (epoch_it % backup_every) == 0:
            print('Backup checkpoint')
            checkpoint_io.save('model_%d.pt' % epoch_it, epoch_it=epoch_it, it=it, loss_val_best=metric_val_best)


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

    return parser.parse_args()


if __name__ == '__main__':
    matplotlib.use('Agg')
    main(parse_args())