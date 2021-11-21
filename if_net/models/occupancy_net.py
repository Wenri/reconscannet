# Occupancy Networks
import torch
import torch.distributions as dist
import torch.nn as nn
from torch.nn import functional as F

from external.common import make_3d_grid
from .encoder import Encoder_Latent, PointNetfeat
from .occ_decoder import DecoderCBatchNorm


class ONet(nn.Module):
    """ Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        encoder_latent (nn.Module): latent encoder network
        p0_z (dist): prior distribution for latent code z
        device (device): torch device
    """

    def __init__(self, cfg, optim_spec=None):
        super(ONet, self).__init__()
        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec

        '''Parameter Configs'''
        decoder_kwargs = {}
        encoder_latent_kwargs = {}
        self.z_dim = cfg['model']['z_dim']
        dim = cfg['data']['dim']
        self.c_dim = cfg['model']['c_dim'] if cfg['data']['skip_propagate'] else 128

        self.encoder_input = PointNetfeat(self.c_dim)

        self.use_cls_for_completion = cfg['data']['use_cls_for_completion']
        if self.use_cls_for_completion:
            self.c_dim += 16
        self.threshold = cfg['test']['threshold']

        '''Module Configs'''
        if self.z_dim != 0:
            self.encoder_latent = Encoder_Latent(dim=dim, z_dim=self.z_dim, c_dim=self.c_dim, **encoder_latent_kwargs)
        else:
            self.encoder_latent = None

        self.decoder = DecoderCBatchNorm(dim=dim, z_dim=self.z_dim, c_dim=self.c_dim, **decoder_kwargs)

        '''Mount mesh generator'''
        if 'generation' in cfg and cfg['generation']['generate_mesh']:
            from .generator import Generator3D
            self.generator = Generator3D(self,
                                         threshold=cfg['test']['threshold'],
                                         resolution0=cfg['generation']['resolution_0'],
                                         upsampling_steps=cfg['generation']['upsampling_steps'],
                                         sample=cfg['generation']['use_sampling'],
                                         refinement_step=cfg['generation']['refinement_step'],
                                         simplify_nfaces=cfg['generation']['simplify_nfaces'],
                                         preprocessor=None)

    def compute_loss(self, input_features_for_completion, input_points_for_completion, input_points_occ_for_completion,
                     voxel_grids, export_shape=False, balance_weight=False, valid_mask=None):
        """
        Compute loss for OccNet
        :param input_features_for_completion (N_B x D): Number of bounding boxes x Dimension of proposal feature.
        :param input_points_for_completion (N_B, N_P, 3): Number of bounding boxes x Number of Points x 3.
        :param input_points_occ_for_completion (N_B, N_P): Corresponding occupancy values.
        :param cls_codes_for_completion (N_B, N_C): One-hot category codes.
        :param export_shape (bool): whether to export a shape voxel example.
        :return:
        """
        device = input_features_for_completion.device
        batch_size = input_features_for_completion.size(0)

        kwargs = {}
        '''Infer latent code z.'''
        if self.z_dim > 0:
            q_z = self.infer_z(input_points_for_completion, input_points_occ_for_completion,
                               input_features_for_completion, device, **kwargs)
            z = q_z.rsample()
            # KL-divergence
            p0_z = self.get_prior_z(self.z_dim, device)
            kl = dist.kl_divergence(q_z, p0_z).sum(dim=-1)
            loss = kl.mean()
        else:
            z = torch.empty(size=(batch_size, 0), device=device)
            loss = 0.

        '''Decode to occupancy voxels.'''
        logits = self(input_points_for_completion, z, input_features_for_completion, voxel_grids)

        num_tot = input_points_occ_for_completion.shape[1]
        if balance_weight:
            num_pos = torch.sum(input_points_occ_for_completion, keepdim=True, dim=-1)
            num_pos[num_pos == 0] = 1e-6
            pos_weight = input_points_occ_for_completion / num_pos
            neg_weight = (1.0 - input_points_occ_for_completion) / (num_tot - num_pos)
            weight = 0.5 * num_tot * (pos_weight + neg_weight) * valid_mask.unsqueeze(-1)
        else:
            weight = valid_mask.unsqueeze(-1).expand(-1, num_tot)

        loss_i = F.binary_cross_entropy_with_logits(
            logits, input_points_occ_for_completion, reduction='none', weight=weight)
        loss = loss + loss_i.sum(-1).mean()

        '''Export Shape Voxels.'''
        if export_shape:
            shape = (16, 16, 16)
            p = make_3d_grid([-0.5 + 1 / 32] * 3, [0.5 - 1 / 32] * 3, shape).to(device)
            p = p.expand(batch_size, *p.size())
            z = self.get_z_from_prior((batch_size,), device, sample=False)
            kwargs = {}
            p_r = self.decode(p, z, input_features_for_completion, **kwargs)

            occ_hat = p_r.probs.view(batch_size, *shape)
            voxels_out = (occ_hat >= self.threshold)
        else:
            voxels_out = None

        return loss, voxels_out

    def forward(self, p, z, c, x=None):
        """
        Performs a forward pass through the network.
        :param input_points_for_completion (tensor): sampled points
        :param input_features_for_completion (tensor): conditioning input
        :param cls_codes_for_completion: class codes for input shapes.
        :param sample (bool): whether to sample for z
        :param kwargs:
        :return:
        """
        return self.decoder(p, z, c)

    def get_z_from_prior(self, size=torch.Size([]), device='cuda', sample=False):
        """ Returns z from prior distribution.

        Args:
            size (Size): size of z
            sample (bool): whether to sample
        """
        p0_z = self.get_prior_z(self.z_dim, device)
        if sample:
            z = p0_z.sample(size)
        else:
            z = p0_z.mean
            z = z.expand(*size, *z.size())

        return z

    def decode(self, input_points_for_completion, z, features, **kwargs):
        """ Returns occupancy probabilities for the sampled points.
        :param input_points_for_completion: points
        :param z: latent code z
        :param features: latent conditioned features
        :return:
        """
        logits = self.decoder(input_points_for_completion, z, features, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def infer_z(self, p, occ, c, device, **kwargs):
        """
        Infers latent code z.
        :param p : points tensor
        :param occ: occupancy values for occ
        :param c: latent conditioned code c
        :param kwargs:
        :return:
        """
        if self.encoder_latent is not None:
            mean_z, softstd_z = self.encoder_latent(p, occ, c, **kwargs)
        else:
            batch_size = p.size(0)
            mean_z = torch.empty(batch_size, 0).to(device)
            softstd_z = torch.empty(batch_size, 0).to(device)

        q_z = dist.Normal(mean_z, softstd_z)
        return q_z

    def get_prior_z(self, z_dim, device):
        """ Returns prior distribution for latent code z.

        Args:
            zdim: dimension of latent code z.
            device (device): pytorch device
        """
        p0_z = dist.Normal(
            torch.zeros(z_dim, device=device),
            torch.ones(z_dim, device=device)
        )

        return p0_z

    def infer_c(self, points, cls_codes_for_completion):
        device = points.device
        input_features_for_completion = self.encoder_input(points)

        if self.use_cls_for_completion:
            cls_codes_for_completion = F.one_hot(cls_codes_for_completion.to(device), num_classes=16).float()
            input_features_for_completion = torch.cat([input_features_for_completion, cls_codes_for_completion], dim=-1)

        return input_features_for_completion
