import torch

import torch
import torch.nn as nn
import torch.nn.functional as F


class ROIToMemoryEncoder(nn.Module):
    """
    Input:
        roi_feat: [B, C, H, W], e.g. [4, 256, 7, 7]
    Output:
        memory_token: [B, 1, C], e.g. [4, 1, 256]
    """
    def __init__(self,
                 dim=256,
                 cfg=None,
                 reduction=4,
                 use_spatial_gate=False):
        super().__init__()
        self.cfg = cfg
        self.use_spatial_gate = use_spatial_gate
        hidden = max(dim // reduction, 32)
        self.out_dim = dim // 4

        # ===== branch for x0 =====
        self.local_fuse_x0 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        self.channel_gate_x0 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        if self.use_spatial_gate:
            self.spatial_gate_x0 = nn.Sequential(
                nn.Conv2d(dim, hidden, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
                nn.Sigmoid()
            )

        self.out_proj_x0 = nn.Sequential(
            nn.Conv2d(dim, self.out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_dim),
            nn.GELU(),
        )

        # ===== branch for x =====
        self.local_fuse_x = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        self.channel_gate_x = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        if self.use_spatial_gate:
            self.spatial_gate_x = nn.Sequential(
                nn.Conv2d(dim, hidden, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
                nn.Sigmoid()
            )

        self.out_proj_x = nn.Sequential(
            nn.Conv2d(dim, self.out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_dim),
            nn.GELU(),
        )

        # ===== fusion gate =====
        fuse_hidden = max(self.out_dim // reduction, 32)
        self.fuse_gate = nn.Sequential(
            nn.Conv2d(self.out_dim * 2, fuse_hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(fuse_hidden),
            nn.GELU(),
            nn.Conv2d(fuse_hidden, self.out_dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(self.out_dim, self.out_dim, kernel_size=3, padding=1, groups=self.out_dim, bias=False),
            nn.BatchNorm2d(self.out_dim),
            nn.GELU(),
            nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_dim),
            nn.GELU(),
        )

    def forward_single_x0(self, roi_feat_x0):
        x0 = self.local_fuse_x0(roi_feat_x0) + roi_feat_x0
        ch_gate_x0 = self.channel_gate_x0(x0)
        x0 = x0 * ch_gate_x0 + roi_feat_x0
        if self.use_spatial_gate:
            sp_gate_x0 = self.spatial_gate_x0(x0)
            x0 = x0 * sp_gate_x0
        x0 = self.out_proj_x0(x0)
        return x0

    def forward_single_x(self, roi_feat_x):
        x = self.local_fuse_x(roi_feat_x) + roi_feat_x
        ch_gate_x = self.channel_gate_x(x)
        x = x * ch_gate_x + roi_feat_x
        if self.use_spatial_gate:
            sp_gate_x = self.spatial_gate_x(x)
            x = x * sp_gate_x
        x = self.out_proj_x(x)
        return x

    def forward(self, roi_feat_x0, roi_feat_x):
        """
        roi_feat_x0: [B, C, H, W]
        roi_feat_x : [B, C, H, W]
        return:     [B, C_out, H, W]
        """
        m0 = self.forward_single_x0(roi_feat_x0)
        mx = self.forward_single_x(roi_feat_x)

        fuse_in = torch.cat([m0, mx], dim=1)
        g = self.fuse_gate(fuse_in)  # [B, C_out, H, W]

        xm = g * mx + (1.0 - g) * m0
        xm = self.refine(xm) + xm
        return xm

def build_memory_encoder(encoder):
    num_channels_enc = encoder.num_channels
    memory_encoder = ROIToMemoryEncoder(dim=num_channels_enc)
    return memory_encoder
