import torch
import torch.distributions as dist
import torch.nn as nn
from torch.nn import functional as F

from external.common import make_3d_grid
from .encoder import Encoder_Latent, PointNetfeat
from .layers import CResnetBlockConv1d, CResnetBlockConv3d


class IFNet(nn.Module):
    def __init__(self, cfg, optim_spec=None, hidden_size=256, n_blocks=5):
        super().__init__()
        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec

        '''Parameter Configs'''
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
            self.encoder_latent = Encoder_Latent(dim=dim, z_dim=self.z_dim, c_dim=self.c_dim)
        else:
            self.encoder_latent = None

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

        # 128**3 res input
        self.conv_in = CResnetBlockConv3d(self.c_dim, 1, 16, 16)
        self.conv_0 = CResnetBlockConv3d(self.c_dim, 16, 32, 32)
        self.conv_1 = CResnetBlockConv3d(self.c_dim, 32, 64, 64)
        self.conv_2 = CResnetBlockConv3d(self.c_dim, 64, 128, 128)
        self.conv_3 = CResnetBlockConv3d(self.c_dim, 128)

        feature_size = (1 + 16 + 32 + 64 + 128 + 128) * 7

        self.fc_p = nn.Conv1d(dim, hidden_size, 1)

        if not self.z_dim == 0:
            self.fc_z = nn.Linear(self.z_dim, hidden_size)

        self.blocks = nn.ModuleList([
            CResnetBlockConv1d(self.c_dim, hidden_size) for _ in range(n_blocks)
        ])
        self.fc_0 = nn.Conv1d(feature_size, hidden_size, 1)
        self.fc_out = nn.Conv1d(hidden_size, 1, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.register_buffer('displacments', torch.Tensor(displacments), persistent=False)

    def forward(self, p, z, c, x=None):
        net = self.fc_p(p.transpose(1, 2))

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        features = net

        if x is not None:
            x = x.transpose(1, 3).unsqueeze(1)

            p_features = features
            p = 2 * p.detach().unsqueeze(1).unsqueeze(1)
            p = torch.cat([p + d for d in self.displacments], dim=2)  # (B,1,7,num_samples,3)
            feature_0 = F.grid_sample(x, p, padding_mode='border')  # out : (B,C (of x), 1,1,sample_num)

            net = self.conv_in(x, c)
            feature_1 = F.grid_sample(net, p, padding_mode='border')  # out : (B,C (of x), 1,1,sample_num)
            net = self.maxpool(net)

            net = self.conv_0(net, c)
            feature_2 = F.grid_sample(net, p, padding_mode='border')  # out : (B,C (of x), 1,1,sample_num)
            net = self.maxpool(net)

            net = self.conv_1(net, c)
            feature_3 = F.grid_sample(net, p, padding_mode='border')  # out : (B,C (of x), 1,1,sample_num)
            net = self.maxpool(net)

            net = self.conv_2(net, c)
            feature_4 = F.grid_sample(net, p, padding_mode='border')
            net = self.maxpool(net)

            net = self.conv_3(net, c)
            feature_5 = F.grid_sample(net, p, padding_mode='border')

            # here every channel corresponds to one feature.

            features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5),
                                 dim=1)  # (B, features, 1,7,sample_num)
            shape = features.shape
            features = features.view(shape[0], shape[1] * shape[3], shape[4])  # (B, featues_per_sample, samples_num)
            features = self.actvn(self.fc_0(features))
            features = features + p_features  # (B, featue_size, samples_num)

        net = features
        for block in self.blocks:
            net = block(net, c)

        net = self.fc_out(net)
        out = net.squeeze(1)

        return out

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

    def decode(self, input_points_for_completion, z, features, voxel_grid):
        """ Returns occupancy probabilities for the sampled points.
        :param input_points_for_completion: points
        :param z: latent code z
        :param features: latent conditioned features
        :return:
        """
        logits = self(input_points_for_completion, z, features, voxel_grid)
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
