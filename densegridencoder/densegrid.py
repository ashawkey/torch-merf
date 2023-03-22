import torch
import torch.nn as nn
import torch.nn.functional as F

from .cuda_gridsample import grid_sample_2d, grid_sample_3d

# MeRF like
class DenseGrid(nn.Module):

    def __init__(self, 
                 output_dim,
                 L_res,
                 H_res,
                 align_corners=True,
                 ):

        super(DenseGrid, self).__init__()

        self.output_dim = output_dim
        self.L_res = L_res
        self.H_res = H_res
        self.align_corners = align_corners
        
        self.L_grid = nn.Parameter(torch.zeros([1, output_dim] + [self.L_res] * 3))
        self.H_planes = nn.Parameter(torch.zeros([3, output_dim] + [self.H_res] * 2))

    def forward(self, xyz, bound=1):

        shape = xyz.shape[:-1]
        xyz_norm = xyz / bound

        L_out = grid_sample_3d(self.L_grid, xyz_norm.view(1, 1, 1, -1, 3).contiguous(), align_corners=self.align_corners).view(self.output_dim, -1)
        H_out = grid_sample_2d(self.H_planes[[0]], xyz_norm[..., [0, 1]].view(1, 1, -1, 2).contiguous(), align_corners=self.align_corners).view(self.output_dim, -1) + \
                grid_sample_2d(self.H_planes[[1]], xyz_norm[..., [1, 2]].view(1, 1, -1, 2).contiguous(), align_corners=self.align_corners).view(self.output_dim, -1) + \
                grid_sample_2d(self.H_planes[[2]], xyz_norm[..., [2, 0]].view(1, 1, -1, 2).contiguous(), align_corners=self.align_corners).view(self.output_dim, -1)

        out = (L_out + H_out).T.reshape(*shape, self.output_dim)

        return out

    def extra_repr(self):
        return f'[DenseGridEncoder] output_dim={self.output_dim}, resolution={self.L_res} grid / {self.H_res} triplane'  