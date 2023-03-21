import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x

# MeRF like
class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 opt,
                 ):

        super().__init__(opt)

        # grid
        self.grid_encoder, self.grid_in_dim = get_encoder("hashgrid", input_dim=3, level_dim=2, num_levels=16, log2_hashmap_size=19, desired_resolution=512)
        self.grid_mlp = MLP(self.grid_in_dim, 8, 64, 2, bias=False)

        # triplane
        # NOTE: per encoder per MLP? or one MLP for all encoders?
        self.planeXY_encoder, self.plane_in_dim = get_encoder("hashgrid", input_dim=2, level_dim=2, num_levels=16, log2_hashmap_size=19, desired_resolution=2048)
        self.planeYZ_encoder, self.plane_in_dim = get_encoder("hashgrid", input_dim=2, level_dim=2, num_levels=16, log2_hashmap_size=19, desired_resolution=2048)
        self.planeXZ_encoder, self.plane_in_dim = get_encoder("hashgrid", input_dim=2, level_dim=2, num_levels=16, log2_hashmap_size=19, desired_resolution=2048)
        self.plane_mlp = MLP(self.plane_in_dim, 8, 64, 2, bias=False)

        # view-dependency
        self.view_encoder, self.view_in_dim = get_encoder('frequency', input_dim=3, multires=4)
        self.view_mlp = MLP(3 + 4 + self.view_in_dim, 3, 16, 3, bias=True)

        # proposal network
        if not self.opt.cuda_ray:
            self.prop_encoders = nn.ModuleList()
            self.prop_mlp = nn.ModuleList()

            # hard coded 2-layer prop network
            prop0_encoder, prop0_in_dim = get_encoder("hashgrid", input_dim=3, level_dim=2, num_levels=5, log2_hashmap_size=17, desired_resolution=128)
            prop0_mlp = MLP(prop0_in_dim, 1, 16, 2, bias=False)
            self.prop_encoders.append(prop0_encoder)
            self.prop_mlp.append(prop0_mlp)

            prop1_encoder, prop1_in_dim = get_encoder("hashgrid", input_dim=3, level_dim=2, num_levels=5, log2_hashmap_size=17, desired_resolution=256)
            prop1_mlp = MLP(prop1_in_dim, 1, 16, 2, bias=False)
            self.prop_encoders.append(prop1_encoder)
            self.prop_mlp.append(prop1_mlp)

    def common_forward(self, x):

        f_grid = self.grid_mlp(self.grid_encoder(x, bound=self.bound))

        f_plane = self.plane_mlp(self.planeXY_encoder(x[..., [0, 1]], bound=self.bound)) + \
                  self.plane_mlp(self.planeYZ_encoder(x[..., [1, 2]], bound=self.bound)) + \
                  self.plane_mlp(self.planeXZ_encoder(x[..., [0, 2]], bound=self.bound))

        f = f_grid + f_plane

        sigma = trunc_exp(f[..., 0])
        diffuse = torch.sigmoid(f[..., 1:4])
        f_specular = torch.sigmoid(f[..., 4:])

        return sigma, diffuse, f_specular

    def forward(self, x, d, shading='full'):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        sigma, diffuse, f_specular = self.common_forward(x)

        d = self.view_encoder(d)
        if shading == 'diffuse':
            color = diffuse
            specular = None
        else: 
            specular = self.view_mlp(torch.cat([diffuse, f_specular, d], dim=-1))
            specular = torch.sigmoid(specular)
            if shading == 'specular':
                color = specular
            else: # full
                color = (specular + diffuse).clamp(0, 1) # specular + albedo

        return {
            'sigma': sigma,
            'color': color,
            'specular': specular,
        }


    def density(self, x, proposal=-1):

        # proposal network
        if proposal >= 0 and proposal < len(self.prop_encoders):
            sigma = trunc_exp(self.prop_mlp[proposal](self.prop_encoders[proposal](x, bound=self.bound)).squeeze(-1))
        # final NeRF
        else:
            sigma, _, _ = self.common_forward(x)

        return {
            'sigma': sigma,
        }
    
    def apply_total_variation(self, lambda_tv):
        self.grid_encoder.grad_total_variation(lambda_tv)
        # self.planeXY_encoder.grad_total_variation(lambda_tv)
        # self.planeXZ_encoder.grad_total_variation(lambda_tv)
        # self.planeYZ_encoder.grad_total_variation(lambda_tv)

    # optimizer utils
    def get_params(self, lr):

        params = []

        params.extend([
            {'params': self.grid_encoder.parameters(), 'lr': lr},
            {'params': self.planeXY_encoder.parameters(), 'lr': lr},
            {'params': self.planeYZ_encoder.parameters(), 'lr': lr}, 
            {'params': self.planeXZ_encoder.parameters(), 'lr': lr},
            {'params': self.grid_mlp.parameters(), 'lr': lr}, 
            {'params': self.plane_mlp.parameters(), 'lr': lr}, 
            {'params': self.view_mlp.parameters(), 'lr': lr}, 
        ])

        if not self.opt.cuda_ray:
            params.extend([
                {'params': self.prop_encoders.parameters(), 'lr': lr},
                {'params': self.prop_mlp.parameters(), 'lr': lr},
            ])

        return params