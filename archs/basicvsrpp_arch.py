"""BasicVSR++ (CVPR 2022) baseline, ported from mmediting's BasicVSRPlusPlus to drop the
mmcv/mmedit dependency. The only mmcv-specific piece was `ModulatedDeformConv2d` /
`modulated_deform_conv2d` (second-order deformable alignment); it is replaced here with
`torchvision.ops.deform_conv2d`, the same modulated-deformable-conv backend already used by
this project's own DCNv2PackFlowGuided modules (see vrt_arch.py / gsmamba_arch.py) -- same
operation, same weight/bias parameter names, different backend. Reuses this project's
basicsr.archs.spynet_arch.SpyNet and basicsr.archs.arch_util helpers, matching the flat
(non mmcv-ConvModule) key layout the official basicvsr_plusplus_reds4 checkpoint expects for
its spynet submodule.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

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


class ModulatedDeformConv2d(nn.Module):
    """Minimal, mmcv-free parameter holder matching mmcv's ModulatedDeformConv2d layout
    (in_channels, out_channels, weight, bias, stride, padding, dilation, groups,
    deform_groups) so that official BasicVSR++ checkpoints load without key remapping."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, deform_groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deform_groups = deform_groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        nn.init.kaiming_uniform_(self.weight, a=1)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)


class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module (BasicVSR++, Eq. 6)."""

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        super().__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )
        self.init_offset()

    def init_offset(self):
        nn.init.constant_(self.conv_offset[-1].weight, 0)
        nn.init.constant_(self.conv_offset[-1].bias, 0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride,
                                              self.padding, self.dilation, mask)


@ARCH_REGISTRY.register()
class BasicVSRPlusPlus(nn.Module):
    """BasicVSR++ network structure. Support either x4 upsampling or same size output.

    Ref: BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and
    Alignment, CVPR 2022.
    """

    def __init__(self, mid_channels=64, num_blocks=7, max_residue_magnitude=10,
                 is_low_res_input=True, spynet_path=None, cpu_cache_length=100):
        super().__init__()
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length

        self.spynet = SpyNet(spynet_path)

        if is_low_res_input:
            self.feat_extract = ResidualBlocksWithInputConv(3, mid_channels, 5)
        else:
            self.feat_extract = nn.Sequential(
                nn.Conv2d(3, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ResidualBlocksWithInputConv(mid_channels, mid_channels, 5))

        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                2 * mid_channels, mid_channels, 3, padding=1, deform_groups=16,
                max_residue_magnitude=max_residue_magnitude)
            self.backbone[module] = ResidualBlocksWithInputConv(
                (2 + i) * mid_channels, mid_channels, num_blocks)

        self.reconstruction = ResidualBlocksWithInputConv(5 * mid_channels, mid_channels, 5)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.is_mirror_extended = False

    def check_if_mirror_extended(self, lqs):
        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lqs):
        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:
            flows_forward = None
        else:
            flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()

        return flows_forward, flows_backward

    def propagate(self, feats, flows, module_name):
        n, t, _, h, w = flows.size()

        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]
            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()

            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                if self.cpu_cache:
                    flow_n1 = flow_n1.cuda()

                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:
                    feat_n2 = feats[module_name][-2]
                    if self.cpu_cache:
                        feat_n2 = feat_n2.cuda()

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    if self.cpu_cache:
                        flow_n2 = flow_n2.cuda()

                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n2)

            feat = [feat_current] + [
                feats[k][idx] for k in feats if k not in ['spatial', module_name]
            ] + [feat_prop]
            if self.cpu_cache:
                feat = [f.cuda() for f in feat]

            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)

            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def upsample(self, lqs, feats):
        outputs = []
        num_outputs = len(feats['spatial'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)
            if self.cpu_cache:
                hr = hr.cuda()

            hr = self.reconstruction(hr)
            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            if self.is_low_res_input:
                hr += self.img_upsample(lqs[:, i, :, :, :])
            else:
                hr += lqs[:, i, :, :, :]

            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()

            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    def forward(self, lqs):
        n, t, c, h, w = lqs.size()

        self.cpu_cache = t > self.cpu_cache_length and lqs.is_cuda

        if self.is_low_res_input:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w), scale_factor=0.25, mode='bicubic').view(n, t, c, h // 4, w // 4)

        self.check_if_mirror_extended(lqs)

        feats = {}
        if self.cpu_cache:
            feats['spatial'] = []
            for i in range(0, t):
                feat = self.feat_extract(lqs[:, i, :, :, :]).cpu()
                feats['spatial'].append(feat)
                torch.cuda.empty_cache()
        else:
            feats_ = self.feat_extract(lqs.view(-1, c, h, w))
            h, w = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h, w)
            feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]

        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            f'The height and width of low-res inputs must be at least 64, but got {h} and {w}.')
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'
                feats[module] = []

                if direction == 'backward':
                    flows = flows_backward
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)

                feats = self.propagate(feats, flows, module)
                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()

        return self.upsample(lqs, feats)
