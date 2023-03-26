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


class Grid(nn.Module):
    def __init__(self, level_dim=2, num_levels=16, log2_hashmap_size=19, desired_resolution=512, output_dim=8, interpolation='linear'):
        super().__init__()
        self.resolution = desired_resolution
        self.encoder, self.in_dim = get_encoder("hashgrid", input_dim=3, level_dim=level_dim, num_levels=num_levels, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution + 1, align_corners=True, interpolation=interpolation)
        self.mlp = MLP(self.in_dim, output_dim, 0, 1, bias=False)
    
    def forward(self, xyz, bound):

        xyz = (xyz + bound) / (2 * bound) # [0, 1]
        f = self.mlp(self.encoder(xyz))
        
        return f

    def grad_total_variation(self, lambda_tv):
        self.encoder.grad_total_variation(lambda_tv)


class Plane(nn.Module):
    def __init__(self, level_dim=2, num_levels=16, log2_hashmap_size=19, desired_resolution=2048, output_dim=8, interpolation='linear'):
        super().__init__()
        self.resolution = desired_resolution
        self.encoder, self.in_dim = get_encoder("hashgrid", input_dim=2, level_dim=level_dim, num_levels=num_levels, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, align_corners=False, interpolation=interpolation)
        self.mlp = MLP(self.in_dim, output_dim, 0, 1, bias=False)
    
    def forward(self, xy, bound):
        
        xy = (xy + bound) / (2 * bound) # [0, 1]
        f = self.mlp(self.encoder(xy))
        return f

    def grad_total_variation(self, lambda_tv):
        self.encoder.grad_total_variation(lambda_tv)        


# MeRF like
class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 opt,
                 ):

        super().__init__(opt)

        # grid
        # if self.opt.use_grid:
        self.grid = Grid(level_dim=2, num_levels=16, log2_hashmap_size=19, desired_resolution=512, output_dim=8)
        
        # triplane
        # if self.opt.use_triplane:
        self.planeXY = Plane(level_dim=2, num_levels=16, log2_hashmap_size=19, desired_resolution=2048, output_dim=8)
        self.planeYZ = Plane(level_dim=2, num_levels=16, log2_hashmap_size=19, desired_resolution=2048, output_dim=8)
        self.planeXZ = Plane(level_dim=2, num_levels=16, log2_hashmap_size=19, desired_resolution=2048, output_dim=8)

        # view-dependency
        self.view_encoder, self.view_in_dim = get_encoder('frequency', input_dim=3, multires=4)
        self.view_mlp = MLP(3 + 4 + self.view_in_dim, 3, 16, 3, bias=True)

        # proposal network
        if not self.opt.cuda_ray:
            self.prop_encoders = nn.ModuleList()
            self.prop_mlp = nn.ModuleList()

            # hard coded 2-layer prop network
            prop0_encoder, prop0_in_dim = get_encoder("hashgrid", input_dim=3, level_dim=2, num_levels=10, log2_hashmap_size=16, desired_resolution=256)
            prop0_mlp = MLP(prop0_in_dim, 1, 16, 2, bias=False)
            self.prop_encoders.append(prop0_encoder)
            self.prop_mlp.append(prop0_mlp)

            prop1_encoder, prop1_in_dim = get_encoder("hashgrid", input_dim=3, level_dim=2, num_levels=10, log2_hashmap_size=16, desired_resolution=512)
            prop1_mlp = MLP(prop1_in_dim, 1, 16, 2, bias=False)
            self.prop_encoders.append(prop1_encoder)
            self.prop_mlp.append(prop1_mlp)

    def common_forward(self, x):
        
        f = 0
        if self.opt.use_grid:
            f_grid = self.quantize_feature(self.grid(x, self.bound))
            f = f + f_grid
        if self.opt.use_triplane:
            f_plane_01 = self.quantize_feature(self.planeXY(x[..., [0, 1]], self.bound))
            f_plane_12 = self.quantize_feature(self.planeYZ(x[..., [1, 2]], self.bound))
            f_plane_02 = self.quantize_feature(self.planeXZ(x[..., [0, 2]], self.bound))
            f = f + f_plane_01 + f_plane_12 + f_plane_02
        
        f_sigma = f[..., 0]
        f_diffuse = f[..., 1:4]
        f_specular = f[..., 4:]

        return f_sigma, f_diffuse, f_specular
    
    def quantize_feature(self, f, baking=False):
        f[..., 0] = self.quantize(f[..., 0], 14, baking)
        f[..., 1:] = self.quantize(f[..., 1:], 7, baking)
        return f
    
    def quantize(self, x, m=7, baking=False):
        # x: in real value, to be quantized in to [-m, m]
        x = torch.sigmoid(x)

        if baking: return 255 * x
        
        x = x + (torch.floor(255 * x + 0.5) / 255 - x).detach()
        x = 2 * m * x - m
        return x

    def forward(self, x, d, shading='full'):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        f_sigma, f_diffuse, f_specular = self.common_forward(x)

        sigma = trunc_exp(f_sigma - 1) # in shader they use exp(x - 1)
        diffuse = torch.sigmoid(f_diffuse)
        f_specular = torch.sigmoid(f_specular)

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

        results = {
            'sigma': sigma,
            'color': color,
            'specular': specular,
        }

        return results


    def density(self, x, proposal=-1):

        # proposal network
        if proposal >= 0 and proposal < len(self.prop_encoders):
            x = (x + self.bound) / (2 * self.bound)
            sigma = trunc_exp(self.prop_mlp[proposal](self.prop_encoders[proposal](x)).squeeze(-1) - 1)
        # final NeRF
        else:
            f_sigma, f_diffuse, f_specular = self.common_forward(x)
            sigma = trunc_exp(f_sigma - 1)

        return {
            'sigma': sigma,
        }
    

    def apply_total_variation(self, lambda_tv):
        if self.opt.use_grid:
            self.grid.grad_total_variation(lambda_tv)
        # self.planeXY.grad_total_variation(lambda_tv * 0.1)
        # self.planeXZ.grad_total_variation(lambda_tv * 0.1)
        # self.planeYZ.grad_total_variation(lambda_tv * 0.1)

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.view_mlp.parameters(), 'lr': lr},
        ]

        if self.opt.use_grid:
            params.extend([
                {'params': self.grid.parameters(), 'lr': lr},
            ])
        
        if self.opt.use_triplane:
            params.extend([
                {'params': self.planeXY.parameters(), 'lr': lr},
                {'params': self.planeYZ.parameters(), 'lr': lr}, 
                {'params': self.planeXZ.parameters(), 'lr': lr},
            ])

        if not self.opt.cuda_ray:
            params.extend([
                {'params': self.prop_encoders.parameters(), 'lr': lr},
                {'params': self.prop_mlp.parameters(), 'lr': lr},
            ])

        return params