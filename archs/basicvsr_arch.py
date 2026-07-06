"""BasicVSR (CVPR 2021) baseline, ported from mmediting's BasicVSRNet to drop the mmcv/mmedit
dependency. Reuses this project's basicsr.archs.spynet_arch.SpyNet (flat conv structure, matches
the key layout of the official basicvsr_reds4 checkpoint) and basicsr.archs.arch_util helpers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.spynet_arch import SpyNet
from basicsr.archs.arch_util import flow_warp, make_layer, ResidualBlockNoBN, default_init_weights
from basicsr.utils.registry import ARCH_REGISTRY


class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front."""

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()
        main = [
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_blocks, num_feat=out_channels),
        ]
        self.main = nn.Sequential(*main)

    def forward(self, feat):
        return self.main(feat)


class PixelShufflePack(nn.Module):
    """Pixel Shuffle upsample layer."""

    def __init__(self, in_channels, out_channels, scale_factor, upsample_kernel):
        super().__init__()
        self.scale_factor = scale_factor
        self.upsample_conv = nn.Conv2d(
            in_channels, out_channels * scale_factor * scale_factor, upsample_kernel,
            padding=(upsample_kernel - 1) // 2)
        default_init_weights(self, 1)

    def forward(self, x):
        x = self.upsample_conv(x)
        return F.pixel_shuffle(x, self.scale_factor)


@ARCH_REGISTRY.register()
class BasicVSR2(nn.Module):
    """BasicVSR network structure for video super-resolution. Support only x4 upsampling.

    Ref: BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond, CVPR 2021.
    """

    def __init__(self, mid_channels=64, num_blocks=30, spynet_path=None):
        super().__init__()
        self.mid_channels = mid_channels

        self.spynet = SpyNet(spynet_path)

        self.backward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)

        self.fusion = nn.Conv2d(mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.is_mirror_extended = False

    def check_if_mirror_extended(self, lrs):
        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lrs):
        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:
            flows_forward = None
        else:
            flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, lrs):
        n, t, c, h, w = lrs.size()
        assert h >= 64 and w >= 64, (
            f'The height and width of inputs should be at least 64, but got {h} and {w}.')

        self.check_if_mirror_extended(lrs)
        flows_forward, flows_backward = self.compute_flow(lrs)

        # backward-time propagation
        outputs = []
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([lrs[:, i, :, :, :], feat_prop], dim=1)
            feat_prop = self.backward_resblocks(feat_prop)
            outputs.append(feat_prop)
        outputs = outputs[::-1]

        # forward-time propagation and upsampling
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            lr_curr = lrs[:, i, :, :, :]
            if i > 0:
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([lr_curr, feat_prop], dim=1)
            feat_prop = self.forward_resblocks(feat_prop)

            out = torch.cat([outputs[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.upsample1(out))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            out += self.img_upsample(lr_curr)
            outputs[i] = out

        return torch.stack(outputs, dim=1)
