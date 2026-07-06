import collections.abc
import math
import torch
import torchvision
import warnings
from distutils.version import LooseVersion
from itertools import repeat
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

# from basicsr.ops.dcn import ModulatedDeformConvPack, modulated_deform_conv
from basicsr.utils import get_root_logger
from einops import rearrange

from basicsr.ops.dcn import ModulatedDeformConvPack, modulated_deform_conv



class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    ``Paper: Delving Deep into Deformable Alignment in Video Super-Resolution``
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')

        if LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0'):
            return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                                 self.dilation, mask)
        else:
            return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, self.groups, self.deformable_groups)



@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


def resize_flow(flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow, size=(output_h, output_w), mode=interp_mode, align_corners=align_corners)
    return resized_flow


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


# class DCNv2Pack(ModulatedDeformConvPack):
#     """Modulated deformable conv for deformable alignment.
#
#     Different from the official DCNv2Pack, which generates offsets and masks
#     from the preceding features, this DCNv2Pack takes another different
#     features to generate offsets and masks.
#
#     Ref:
#         Delving Deep into Deformable Alignment in Video Super-Resolution.
#     """
#
#     def forward(self, x, feat):
#         out = self.conv_offset(feat)
#         o1, o2, mask = torch.chunk(out, 3, dim=1)
#         offset = torch.cat((o1, o2), dim=1)
#         mask = torch.sigmoid(mask)
#
#         offset_absmean = torch.mean(torch.abs(offset))
#         if offset_absmean > 50:
#             logger = get_root_logger()
#             logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')
#
#         if LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0'):
#             return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
#                                                  self.dilation, mask)
#         else:
#             return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
#                                          self.dilation, self.groups, self.deformable_groups)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [low, up], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution.

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# From PyTorch
def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple



def warp_features_with_flow(x_unpatched, optical_flow):
    """
    x_unpatched: Tensor [B, T, C, H, W]
    optical_flow: Tensor [B, T, T, 2, H', W']
        flow[i, j] is flow from frame i to j
    Returns:
        warped_feats: Tensor [B, T, T, C, H, W]
    """
    B, T, C, H, W = x_unpatched.shape
    _, _, _, _, Hf, Wf = optical_flow.shape

    # If needed, interpolate features to match flow resolution
    if H != Hf or W != Wf:
        x_unpatched = F.interpolate(
            x_unpatched.view(B * T, C, H, W), size=(Hf, Wf), mode='bilinear', align_corners=False
        ).view(B, T, C, Hf, Wf)

    warped = torch.zeros(B, T, T, C, Hf, Wf, device=x_unpatched.device)

    for i in range(T):
        for j in range(T):
            # flow: from i to j → warp j to align to i
            flow = optical_flow[:, i, j]  # [B, 2, Hf, Wf]
            feat = x_unpatched[:, j]     # [B, C, Hf, Wf]

            # Create normalized grid for warping
            B_, _, H_, W_ = flow.shape
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H_, device=flow.device),
                torch.arange(W_, device=flow.device),
                indexing='ij'
            )
            grid = torch.stack((grid_x, grid_y), dim=0).float()  # [2, H, W]
            grid = grid.unsqueeze(0).expand(B_, -1, -1, -1)  # [B, 2, H, W]

            vgrid = grid + flow  # apply flow
            # Normalize to [-1, 1]
            vgrid[:, 0] = 2.0 * vgrid[:, 0] / max(W_ - 1, 1) - 1.0
            vgrid[:, 1] = 2.0 * vgrid[:, 1] / max(H_ - 1, 1) - 1.0

            vgrid = vgrid.permute(0, 2, 3, 1)  # [B, H, W, 2]

            warped_feat = F.grid_sample(feat, vgrid, align_corners=True)
            warped[:, i, j] = warped_feat

    return warped  # shape: [B, T, T, C, Hf, Wf]



def flow_warp_custom(x, x_size, optical_flow_forward, optical_flow_backward, patch_size=8):
    """
    Warps all frames to each reference frame using multi-step flow composition.
    
    Args:
        x: [B, T, H*W, C] or [B, T, H, W, C]
        x_size: (H, W)
        optical_flow_forward: list of [1, 2, H, W]
        optical_flow_backward: list of [1, 2, H, W]
        patch_size: int
    Returns:
        warped: [B, T, T, C, H, W]
    """
    B, T, _, C = x.shape
    H, W = x_size
    x = x.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3).contiguous()  # [B, T, C, H, W]
    device = x.device

    warped = []

    for ref_idx in range(T):
        warped_t = []
        for tgt_idx in range(T):
            ref = x[:, ref_idx]
            tgt = x[:, tgt_idx]

            if ref_idx == tgt_idx:
                warped_t.append(ref.unsqueeze(1))  # 그대로 사용
                continue

            # ---- get_flow_chain logic ----
            flow_chain = []
            if tgt_idx < ref_idx:
                for i in reversed(range(tgt_idx, ref_idx)):
                    flow_chain.append(('b', i))
            else:
                for i in range(ref_idx, tgt_idx):
                    flow_chain.append(('f', i))

            # ---- flow_align_patch_chain logic ----
            flow_accum = torch.zeros_like(ref[:, :2])  # [B, 2, H, W]
            for direction, idx in flow_chain:
                flow = optical_flow_backward[idx] if direction == 'b' else optical_flow_forward[idx]

                # ---- add_flow logic ----
                grid_y, grid_x = torch.meshgrid(
                    torch.arange(H, device=device),
                    torch.arange(W, device=device),
                    indexing='ij'
                )
                grid = torch.stack((grid_x, grid_y), dim=0).float()  # [2, H, W]
                displaced = grid + flow_accum[0]
                norm_x = 2.0 * displaced[0] / (W - 1) - 1.0
                norm_y = 2.0 * displaced[1] / (H - 1) - 1.0
                norm_grid = torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0)  # [1, H, W, 2]

                sampled = F.grid_sample(flow, norm_grid, align_corners=True)
                flow_accum = flow_accum + sampled

            # ---- flow_align_patch logic ----
            assert H % patch_size == 0 and W % patch_size == 0
            num_y, num_x = H // patch_size, W // patch_size
            N = num_y * num_x

            # 평균 flow per patch
            flow_patches = flow_accum.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            flow_patches = flow_patches.reshape(B, 2, num_y, num_x, patch_size, patch_size)
            flow_avg = flow_patches.mean(dim=(-2, -1))  # [B, 2, ny, nx]

            # expand to each patch
            flow_avg = flow_avg.permute(0, 2, 3, 1).reshape(B * N, 2, 1, 1)
            flow_broadcast = flow_avg.expand(-1, -1, patch_size, patch_size)

            # 각 패치 위치의 grid 만들기
            grid_y, grid_x = torch.meshgrid(
                torch.arange(patch_size, device=device),
                torch.arange(patch_size, device=device),
                indexing='ij'
            )
            base_grid = torch.stack((grid_x, grid_y), dim=-1)  # [p, p, 2]

            offset_y = torch.arange(0, H, patch_size, device=device).repeat_interleave(num_x)
            offset_x = torch.arange(0, W, patch_size, device=device).repeat(num_y)
            patch_offsets = torch.stack((offset_x, offset_y), dim=1)  # [N, 2]

            full_base_grid = base_grid.unsqueeze(0).expand(B * N, -1, -1, -1)
            full_offsets = patch_offsets.repeat(B, 1).reshape(B * N, 1, 1, 2).float()
            full_grid = full_base_grid + full_offsets  # [B*N, 8, 8, 2]

            displaced_grid = full_grid + flow_broadcast.permute(0, 2, 3, 1)  # [B*N, 8, 8, 2]

            # 정규화 [-1, 1]
            norm_x = 2.0 * displaced_grid[..., 0] / (W - 1) - 1.0
            norm_y = 2.0 * displaced_grid[..., 1] / (H - 1) - 1.0
            normalized_grid = torch.stack((norm_x, norm_y), dim=-1)  # [B*N, 8, 8, 2]

            # 마스크 생성
            x_valid = (normalized_grid[..., 0] >= -1.0) & (normalized_grid[..., 0] <= 1.0)
            y_valid = (normalized_grid[..., 1] >= -1.0) & (normalized_grid[..., 1] <= 1.0)
            mask = (x_valid & y_valid).float().unsqueeze(1)  # [B*N, 1, 8, 8]

            mask = mask.reshape(B, num_y, num_x, 1, patch_size, patch_size)
            mask = mask.permute(0, 3, 1, 4, 2, 5).contiguous()
            mask_full = mask.reshape(B, 1, H, W)

            # warp 수행
            tgt_exp = tgt.repeat_interleave(N, dim=0)  # [B*N, C, H, W]
            warped_patches = F.grid_sample(tgt_exp, normalized_grid, mode='nearest', padding_mode='zeros', align_corners=True)
            warped_patches = warped_patches.reshape(B, num_y, num_x, C, patch_size, patch_size)
            warped_patches = warped_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
            warped_full = warped_patches.reshape(B, C, H, W)

            # mask 처리
            warped_full = warped_full * mask_full + ref * (1 - mask_full)
            warped_t.append(warped_full.unsqueeze(1))  # [B, 1, C, H, W]

        warped_t = torch.cat(warped_t, dim=1)      # [B, T, C, H, W]
        warped.append(warped_t.unsqueeze(1))       # [B, 1, T, C, H, W]

    warped = torch.cat(warped, dim=1)              # [B, T, T, C, H, W]
    return warped



def flow_warp_all_frames(x, x_size, params):

    B, T, _, C = x.shape
    x = x.reshape(B, T, x_size[0], x_size[1], C).permute(0,1,4,2,3).contiguous()  # [B, T, C, H, W]
    
    warped = []

    for ref_idx in range(T):
        warped_t = []
        for tgt_idx in range(T):
            ref = x[:, ref_idx]  # [B, C, H, W]
            tgt = x[:, tgt_idx]
            if ref_idx == tgt_idx:
                warped_t.append(ref.unsqueeze(1))  # [B, 1, C, H, W]
            else:
                flow_chain = get_flow_chain(ref_idx, tgt_idx)
                warped_frame, _ = flow_align_patch_chain(ref, tgt, flow_chain, params)
                    
                warped_t.append(warped_frame.unsqueeze(1))  # [B, 1, C, H, W]

        warped_t = torch.cat(warped_t, dim=1)  # [B, T, C, H, W] (tgt frames warped to ref_idx)
        warped.append(warped_t.unsqueeze(1))  # [B, 1, T, C, H, W]
    
    
    warped = torch.cat(warped, dim=1)  # [B, T, T, C, H, W]
    
    # self.visualize_flow_warp_all(warped)       

    return warped


def get_flow_chain(ref_idx, target_idx):
    """
    Returns a list of flows to apply sequentially to go from target_idx to ref_idx.
    Each flow has direction ('f' or 'b') and index.

    Ex) ref=0, target=2 → return [('b', 1), ('b', 0)]
    """
    chain = []
    if target_idx < ref_idx:
        for i in reversed(range(target_idx, ref_idx)):
            chain.append(('b', i))  # flow from target → ref
    else:
        for i in range(ref_idx, target_idx):
            chain.append(('f', i))  # forward from future to ref

    return chain

def flow_align_patch_chain(ref_img, target_img, flow_chain, params):
    """
    Args:
        ref_img: [1, C, H, W] 기준 프레임
        target_img: [1, C, H, W] 워핑할 프레임
        flow_chain: 리스트 [('b', 1), ('b', 0), ...]
    Returns:
        warped: 기준 ref에 맞춰 warp된 target_img
    """
    
    optical_flow_backward = params['optical_flow_backward']
    optical_flow_forward = params['optical_flow_forward']
    
    flow_accum = torch.zeros_like(ref_img[:, :2])  # 초기 flow: [1, 2, H, W]
    
    for direction, idx in flow_chain:
        flow = (
            optical_flow_backward[idx]
            if direction == 'b'
            else optical_flow_forward[idx]
        )  # [1, 2, H, W]

        # 현재 flow_accum에 flow를 따라 한 스텝 이동
        flow_accum = add_flow(flow_accum, flow)

    warped, mask = flow_align_patch(ref_img, target_img, flow_accum)
    return warped, mask





def flow_align_patch(cur_frame, next_frame, flow, patch_size=8, interpolation='nearest', padding_mode='zeros', align_corners=True):
    
    B, C, H, W = cur_frame.shape
    ref_patch = cur_frame
    assert H % patch_size == 0 and W % patch_size == 0, "Image must be divisible by patch size"

    num_y, num_x = H // patch_size, W // patch_size
    N = num_y * num_x

    # 1. 쪼갠 패치: [B, C, H//p, W//p, p, p] → [B*N, C, p, p]
    patches = cur_frame.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)  # [B, C, ny, nx, p, p]
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().reshape(B * N, C, patch_size, patch_size)

    # 2. flow 평균 추출
    flow_patches = flow.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)  # [B, 2, ny, nx, p, p]
    flow_patches = flow_patches.contiguous().reshape(B, 2, num_y, num_x, patch_size, patch_size)
    flow_avg = flow_patches.mean(dim=(-2, -1))  # [B, 2, ny, nx]

    # 3. broadcast flow: [B, 2, ny, nx] → [B*N, 2, 8, 8]
    flow_avg = flow_avg.permute(0, 2, 3, 1)             # [B, ny, nx, 2]
    flow_avg = flow_avg.reshape(B * N, 2, 1, 1)         # [B*N, 2, 1, 1]
    flow_broadcast = flow_avg.expand(-1, -1, patch_size, patch_size)  # [B*N, 2, 8, 8]

    # 4. 각 패치 위치 grid
    device = cur_frame.device
    grid_y, grid_x = torch.meshgrid(
        torch.arange(patch_size, device=device),
        torch.arange(patch_size, device=device),
        indexing='ij'
    )  # [8, 8]
    base_grid = torch.stack((grid_x, grid_y), dim=-1)  # [8, 8, 2]

    offset_y = torch.arange(0, H, patch_size, device=device).repeat_interleave(num_x)
    offset_x = torch.arange(0, W, patch_size, device=device).repeat(num_y)
    patch_offsets = torch.stack((offset_x, offset_y), dim=1)  # [N, 2]

    full_base_grid = base_grid.unsqueeze(0).expand(B * N, -1, -1, -1)  # [B*N, 8, 8, 2]
    full_offsets = patch_offsets.repeat(B, 1).reshape(B * N, 1, 1, 2).float()
    full_grid = full_base_grid + full_offsets  # [B*N, 8, 8, 2]

    # 5. flow 더해서 이동된 위치
    flow_grid = flow_broadcast.permute(0, 2, 3, 1)  # [B*N, 8, 8, 2]
    displaced_grid = full_grid + flow_grid  # [B*N, 8, 8, 2]

    # 6. 정규화 [-1, 1]
    norm_x = 2.0 * displaced_grid[..., 0] / (W - 1) - 1.0
    norm_y = 2.0 * displaced_grid[..., 1] / (H - 1) - 1.0
    normalized_grid = torch.stack((norm_x, norm_y), dim=-1)  # [B*N, 8, 8, 2]
    
    x_valid = (normalized_grid[..., 0] >= -1.0) & (normalized_grid[..., 0] <= 1.0)
    y_valid = (normalized_grid[..., 1] >= -1.0) & (normalized_grid[..., 1] <= 1.0)
    mask = (x_valid & y_valid).float().unsqueeze(1) 
    mask_patches = mask.reshape(B, num_y, num_x, 1, patch_size, patch_size)

    # 2. permute to [B, 1, num_y, pH, num_x, pW]
    mask_patches = mask_patches.permute(0, 3, 1, 4, 2, 5).contiguous()

    # 3. 합쳐서 full mask: [B, 1, H, W]
    mask_full = mask_patches.reshape(B, 1, H, W)
    

    # 7. grid_sample from next frame
    next_frame_expanded = next_frame.repeat_interleave(N, dim=0)  # [B*N, C, H, W]
    warped_patches = F.grid_sample(
        next_frame_expanded,
        normalized_grid,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners
    )  # [B*N, C, 8, 8]

    # 8. 다시 합치기
    warped_patches = warped_patches.reshape(B, num_y, num_x, C, patch_size, patch_size)
    warped_patches = warped_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
    warped_full = warped_patches.reshape(B, C, H, W)

    warped_full = warped_full * mask_full  + ref_patch * (1 - mask_full)
    return warped_full, mask_full



def add_flow(flow1, flow2):
    """
    Compose two flows: flow1 followed by flow2.
    Args:
        flow1, flow2: [B, 2, H, W]
    Returns:
        composed: [B, 2, H, W]
    """
    B, _, H, W = flow1.shape

    # grid 만들기
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=flow1.device),
        torch.arange(W, device=flow1.device),
        indexing='ij'
    )
    grid = torch.stack((grid_x, grid_y), dim=0).float()  # [2, H, W]
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H, W]

    displaced = grid + flow1  # [B, 2, H, W]

    # Normalize to [-1, 1]
    norm_x = 2.0 * displaced[:, 0, :, :] / (W - 1) - 1.0
    norm_y = 2.0 * displaced[:, 1, :, :] / (H - 1) - 1.0
    norm_grid = torch.stack((norm_x, norm_y), dim=-1)  # [B, H, W, 2]

    # Sample flow2 at displaced locations
    flow2_sampled = F.grid_sample(flow2, norm_grid, align_corners=True)  # [B, 2, H, W]
    composed = flow1 + flow2_sampled
    return composed




def patch_warp(x, params, patch_size=16):
    
    B, T, C, H, W = x.shape
    device = x.device
    
    optical_flow_forward = params['optical_flow_forward']
    optical_flow_backward = params['optical_flow_backward']
    
    
    all_flows = [[None for _ in range(T)] for _ in range(T)]
    for ref in range(T):
        for tgt in range(T):
            if ref == tgt:
                continue

            flow_chain = []
            if tgt < ref:
                for i in reversed(range(tgt, ref)):
                    flow_chain.append(('b', i))
            else:
                for i in range(ref, tgt):
                    flow_chain.append(('f', i))

            flow_accum = torch.zeros((B, 2, H, W), device=device)
            for direction, idx in flow_chain:
                flow = optical_flow_backward[idx - 1] if direction == 'b' else optical_flow_forward[idx]

                # Compose flow
                grid_y, grid_x = torch.meshgrid(
                    torch.arange(H, device=device),
                    torch.arange(W, device=device),
                    indexing='ij'
                )
                grid = torch.stack((grid_x, grid_y), dim=0).float() # [2, 64, 64]
                grid = grid.unsqueeze(0).repeat(B, 1, 1, 1) # [1, 2, 64, 64]
                displaced = grid + flow_accum
                
                norm_x = 2.0 * displaced[:,0,:,:] / (W - 1) - 1.0
                norm_y = 2.0 * displaced[:,1,:,:] / (H - 1) - 1.0
                norm_grid = torch.stack((norm_x, norm_y), dim=-1)
                sampled = F.grid_sample(flow, norm_grid, align_corners=True)
                flow_accum = flow_accum + sampled   # [1, 2, 64, 64]

            all_flows[ref][tgt] = flow_accum

    warped = []
    
    for ref_idx in range(T):
        ref = x[:, ref_idx]  # [B, C, H, W]
        warped_t = [ref.unsqueeze(1)]  # [B, 1, C, H, W] ref가 반드시 맨 앞에 오도록

        for tgt_idx in range(T):
            if tgt_idx == ref_idx:
                continue

            tgt = x[:, tgt_idx]
            flow = all_flows[ref_idx][tgt_idx]  # [B, 2, H, W]

            # Patch 기반 warping
            num_y, num_x = H // patch_size, W // patch_size
            N = num_y * num_x

            flow_patches = flow.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            flow_patches = flow_patches.reshape(B, 2, num_y, num_x, patch_size, patch_size)
            flow_avg = flow_patches.mean(dim=(-2, -1))  # [B, 2, ny, nx]
            flow_avg = flow_avg.permute(0, 2, 3, 1).reshape(B * N, 2, 1, 1)
            flow_broadcast = flow_avg.expand(-1, -1, patch_size, patch_size)

            grid_y, grid_x = torch.meshgrid(
                torch.arange(patch_size, device=device),
                torch.arange(patch_size, device=device),
                indexing='ij'
            )
            base_grid = torch.stack((grid_x, grid_y), dim=-1)
            offset_y = torch.arange(0, H, patch_size, device=device).repeat_interleave(num_x)
            offset_x = torch.arange(0, W, patch_size, device=device).repeat(num_y)
            patch_offsets = torch.stack((offset_x, offset_y), dim=1)
            full_base_grid = base_grid.unsqueeze(0).expand(B * N, -1, -1, -1)
            full_offsets = patch_offsets.repeat(B, 1).reshape(B * N, 1, 1, 2).float()
            full_grid = full_base_grid + full_offsets
            displaced_grid = full_grid + flow_broadcast.permute(0, 2, 3, 1)

            norm_x = 2.0 * displaced_grid[..., 0] / (W - 1) - 1.0
            norm_y = 2.0 * displaced_grid[..., 1] / (H - 1) - 1.0
            normalized_grid = torch.stack((norm_x, norm_y), dim=-1)

            x_valid = (normalized_grid[..., 0] >= -1.0) & (normalized_grid[..., 0] <= 1.0)
            y_valid = (normalized_grid[..., 1] >= -1.0) & (normalized_grid[..., 1] <= 1.0)
            mask = (x_valid & y_valid).float().unsqueeze(1)
            mask = mask.reshape(B, num_y, num_x, 1, patch_size, patch_size)
            mask = mask.permute(0, 3, 1, 4, 2, 5).contiguous()
            mask_full = mask.reshape(B, 1, H, W)

            tgt_exp = tgt.repeat_interleave(N, dim=0)
            warped_patches = F.grid_sample(tgt_exp, normalized_grid, mode='nearest', padding_mode='zeros', align_corners=True)
            warped_patches = warped_patches.reshape(B, num_y, num_x, C, patch_size, patch_size)
            warped_patches = warped_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
            warped_full = warped_patches.reshape(B, C, H, W)
            warped_full = warped_full * mask_full + ref * (1 - mask_full)

            warped_t.append(warped_full.unsqueeze(1))  # [B, 1, C, H, W]

        warped_row = torch.cat(warped_t, dim=1)        # [B, T, C, H, W] (ref + warped others)
        warped.append(warped_row.unsqueeze(1))   # [B, 1, T, C, H, W]

    warped = torch.cat(warped, dim=1)  # [B, T, T, C, H, W]
    
    return warped




# ↓ → ↑
def s_shape_indices(H, W):
    # Generates S-shape scanning order for H×W grid
    indices = []
    for j in range(W):
        col = list(range(H)) if j % 2 == 0 else list(reversed(range(H)))
        for i in col:
            indices.append((i, j))
    return indices

def s_shape_indices_unfold(h, w):
    """Generate S-shape scan indices for h x w grid"""
    indices = []
    for i in range(h):
        row = list(range(w))
        if i % 2 == 1:
            row.reverse()
        for j in row:
            indices.append((i, j))
    return indices  # List of (i, j)

# → ↓ ←
def s_shape_flat_indices(H, W):
    indices = []
    for i in range(H):
        row = list(range(W)) if i % 2 == 0 else list(reversed(range(W)))
        for j in row:
            indices.append(i * W + j)
    return indices


def s_shape_temporal_unflatten(out, T, H, W, patch_size=8):
    """
    Args:
        out: Tensor of shape [B, C, T * num_patches * patch_area]
        T: number of frames
        H, W: original height and width
        patch_size: size of patch used during flatten
    Returns:
        x: Tensor of shape [B, T, C, H, W]
    """
    B, C, L = out.shape
    ph, pw = H // patch_size, W // patch_size
    patch_area = patch_size * patch_size

    assert L == T * ph * pw * patch_area, "Mismatch in sequence length"

    # Step 1: recover patch list
    patch_len = patch_area * T
    patches = out.chunk(ph * pw, dim=-1)  # list of [B, C, patch_len]

    # Step 2: reorder patches back from S-shape order
    s_patch_idx = s_shape_indices(ph, pw)
    patch_map = [[None for _ in range(pw)] for _ in range(ph)]

    for i, (ph_i, pw_i) in enumerate(s_patch_idx):
        patch = patches[i]

        # Flip back if it was reversed during flatten
        if pw_i % 2 == 1:
            patch = patch.flip(-1)

        patch_map[ph_i][pw_i] = patch  # place it back to original location

    # Step 3: stack patch map → [B, C, ph, pw, patch_len]
    patch_tensor = torch.stack([torch.stack(row, dim=2) for row in patch_map], dim=2)  # [B, C, ph, pw, patch_len]

    # Step 4: reshape → [B, ph, pw, C, patch_area, T]
    patch_tensor = patch_tensor.permute(0, 2, 3, 1, 4).contiguous()  # [B, ph, pw, C, patch_len]
    patch_tensor = patch_tensor.view(B, ph, pw, C, patch_area, T)

    # Step 5: un-S-shape the pixel order
    rev_s_pix_idx = torch.tensor(s_shape_flat_indices(patch_size, patch_size)).argsort()
    patch_tensor = patch_tensor[:, :, :, :, rev_s_pix_idx, :]  # [B, ph, pw, C, patch_area, T]

    # Step 6: reshape → [B, ph, pw, C, h, w, T]
    patch_tensor = patch_tensor.view(B, ph, pw, C, patch_size, patch_size, T)

    # Step 7: rearrange back to [B, T, C, H, W]
    x = rearrange(patch_tensor, 'b ph pw c h w t -> b t c (ph h) (pw w)')

    return x


def s_shape_temporal_flatten(x, patch_size=8):
    """
    Args:
        x: Tensor of shape [B, T, C, H, W]
        patch_size: size of square patch (default 2 for toy example)
    Returns:
        out: Tensor of shape [B, C, T * num_patches * patch_area] flattened in S-shape (temporal-first)
    """
    B, T, C, H, W = x.shape
    ph, pw = H // patch_size, W // patch_size

    # Step 1: patchify with temporal last
    patch_x = rearrange(x, 'b t c (ph h) (pw w) -> b ph pw c h w t', ph=ph, pw=pw)

    # Step 2: flatten spatial inside patch
    patch_x = rearrange(patch_x, 'b ph pw c h w t -> b ph pw c (h w) t')

    # Step 3: apply S-shape inside patch
    s_pix_idx = s_shape_flat_indices(patch_size, patch_size)
    patch_x = patch_x[..., s_pix_idx, :]  # [B, ph, pw, C, patch_area, T]

    # Step 4: temporal-first flatten
    patch_x = rearrange(patch_x, 'b ph pw c p t -> b c ph pw (p t)')

    # Step 5: reorder patches in S-shape (column-wise if needed)
    s_patch_idx = s_shape_indices(ph, pw)  # or row-wise: s_shape_indices(ph, pw)

    patches = []
    # patches = [patch_x[:, :, ph_i, pw_i, :] for ph_i, pw_i in s_patch_idx]
    for ph_i, pw_i in s_patch_idx:
        patch = patch_x[:, :, ph_i, pw_i, :]  # [B, C, patch_len]

        # Flip patch content if column is odd
        if pw_i % 2 == 1:
            patch = patch.flip(-1)

        patches.append(patch)
    out = torch.cat(patches, dim=-1)  # [B, C, total_seq_len]

    return out







def patch_temporal_s_scan(x_unfolded, patch_size=16):
    """
    x_unfolded: [B, Seg, T, C, H, W]
    Returns:
        flattened tensor: [B * Seg * num_patches, C, T * patch_area]
    """
    
    B, Seg, T, C, H, W = x_unfolded.shape
    assert H % patch_size == 0 and W % patch_size == 0, "Image size must be divisible by patch size"

    ph, pw = H // patch_size, W // patch_size  # 4 x 4
    num_patches = ph * pw

    outputs = []

    # Iterate over Segments
    for seg_idx in range(Seg):
        x_seg = x_unfolded[:, seg_idx]  # [B, T, C, H, W]

        # Step 1: reshape into patches
        patches = rearrange(
            x_seg,  # [B, T, C, H, W]
            'b t c (ph p1) (pw p2) -> b ph pw t c p1 p2',
            ph=ph, pw=pw
        )  # [B, ph, pw, T, C, patch_h, patch_w]

        # Step 2: for each patch (in S-shape), extract T-length patch sequence and flatten
        for i, j in s_shape_indices(ph, pw):
            patch_seq = patches[:, i, j]  # [B, T, C, patch_h, patch_w]
            patch_seq = rearrange(patch_seq, 'b t c h w -> b c t (h w)')  # [B, C, T, patch_area]
            patch_seq = rearrange(patch_seq, 'b c t p -> b c (t p)')  # [B, C, T * patch_area]
            outputs.append(patch_seq)

    out = torch.cat(outputs, dim=0)  # [B * Seg * num_patches, C, T * patch_area]
    return out


def patch_temporal_s_fold(out_y, seg=8, T=8, patch_size=16, H=64, W=64):
    """
    out_y: [seg * num_patches, C, T * patch_area]  = [128, 348, 2048]
    Returns:
        x_folded: [1, seg, T, C, H, W]
    """
    B = 1
    C = out_y.shape[1]
    ph, pw = H // patch_size, W // patch_size  # 4x4
    patch_area = patch_size * patch_size

    x_folded = torch.zeros((B, seg, T, C, H, W), dtype=out_y.dtype, device=out_y.device)

    idx = 0
    for seg_idx in range(seg):
        # 임시 공간 [T, C, H, W]
        seg_frames = torch.zeros((T, C, H, W), dtype=out_y.dtype, device=out_y.device)
        s_patch_idx = s_shape_indices_unfold(ph, pw)

        for i, j in s_patch_idx:
            patch_seq = out_y[idx]  # [C, T * patch_area]
            patch_seq = patch_seq.view(C, T, patch_area)  # [C, T, 256]
            patch_seq = patch_seq.permute(1, 0, 2).contiguous()  # [T, C, patch_area]
            patch_seq = patch_seq.view(T, C, patch_size, patch_size)  # [T, C, 16, 16]

            # Flip back if column was odd (undo reverse)
            if j % 2 == 1:
                patch_seq = patch_seq.flip(-1)

            h_start, w_start = i * patch_size, j * patch_size
            seg_frames[:, :, h_start:h_start+patch_size, w_start:w_start+patch_size] = patch_seq

            idx += 1

        x_folded[:, seg_idx] = seg_frames  # [T, C, H, W]

    return x_folded  # [1, seg, T, C, H, W]





def i_shape_flat_indices(H, W):
    indices = i_shape_indices(H, W, H)
    flat = []
    for i, j in indices:
        flat.append(i * W + j)

    if(H == 2):
        flat = [0, 2, 3, 1]
    elif(H == 1):
        flat = [0]
        
        
    return flat
 
 
def i_shape_indices(H, W, patch_size=8):
    """
    [(0, 0), (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 3), (2, 2), (2, 1), (2, 0),
    (3, 0), (3, 1), (3, 2), (3, 3),
    (4, 3), (4, 2), (4, 1), (4, 0),
    (5, 0), (5, 1), (5, 2), (5, 3),
    (6, 3), (6, 2), (6, 1), (6, 0),
    (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7),
    (6, 7), (6, 6), (6, 5), (6, 4),
    (5, 4), (5, 5), (5, 6), (5, 7),
    (4, 7), (4, 6), (4, 5), (4, 4),
    (3, 4), (3, 5), (3, 6), (3, 7),
    (2, 7), (2, 6), (2, 5), (2, 4),
    (1, 4), (1, 5), (1, 6), (1, 7),
    (0, 7), (0, 6), (0, 5), (0, 4), (0, 3), (0, 2), (0, 1)]
    """  
 
    indices = []
    i, j = 0, 0
    half = patch_size // 2
    downward = True
    horizontal_direction = 1  # 1: right, -1: left
   
    indices.append((i, j))
    i += 1
   
    while len(indices) + 1 < H * W:
        indices.append((i, j))
        if downward:
            if i < H - 1:
                for _ in range(half - 1):
                    j += horizontal_direction
                    indices.append((i, j))
                horizontal_direction *= -1
                i += 1
            else:    
                for _ in range(W - 1):
                    j += horizontal_direction
                    indices.append((i, j))
                horizontal_direction *= -1
                downward = False
                i -= 1
        else:
            if i > 0:
                for _ in range(half - 1):
                    if 0 <= j + horizontal_direction < W:
                        j += horizontal_direction
                        indices.append((i, j))
                horizontal_direction *= -1
                i -= 1
            else:
                for _ in range(W - 2):
                    j += horizontal_direction
                    indices.append((i, j))
                horizontal_direction *= -1
                downward = True
                i += 1
   
    return indices
 
def i_shape_temporal_flatten(x, patch_size=8):
    """
    Args:
        x: Tensor of shape [B, T, C, H, W]
        patch_size: size of square patch (default 2 for toy example)
    Returns:
        out: Tensor of shape [B, C, T * num_patches * patch_area] flattened in S-shape (temporal-first)
    """
    B, T, C, H, W = x.shape     # [8, 6, 256, 64, 64]
    ph, pw = H // patch_size, W // patch_size   # 16, 16
   
    # Step 1: patchify with spatial last
    patch_x = rearrange(x, 'b t c (ph h) (pw w) -> b ph pw t c h w', ph=ph, pw=pw)
 
    # Step 2: flatten spatial inside patch
    patch_x = rearrange(patch_x, 'b ph pw t c h w -> b ph pw t c (h w)')    # [8, 16, 16, 6, 256, 16]
   
    # Step 3: apply I-shape inside patch
    i_pix_idx = i_shape_flat_indices(patch_size, patch_size)
    patch_x = patch_x[..., i_pix_idx]  # [B, ph, pw, T, C, patch_area]
   
    # Step 4: I-shape order for patches
    # if(pw == 2):
    #     i_patch_idx = [(0,0), (1,0), (1,1), (0,1)]
    # elif(pw == 1):
    #     i_patch_idx = [(0,0)]
    # else:
    #     i_patch_idx = i_shape_indices(ph, pw, ph)
    i_patch_idx = s_shape_indices(ph, pw)
    

    patches = []
    for ph_i, pw_i in i_patch_idx:
        patch = patch_x[:, ph_i, pw_i, :, :, :] # [8, 6, 256, 16]
        patch = rearrange(patch, 'b t c p -> b c (t p)')    # [8, 256, 96]
       
        patches.append(patch)
   
    out = torch.cat(patches, dim=-1)  # [B, C, total_seq_len]
    return out
 
 
 
 
def i_shape_temporal_unflatten(out, B, T, C, H, W, patch_size=8):
    """
    Args:
        flat: [B, C, total_seq_len] (flattened from i_shape_temporal_flatten)
        B, T, C, H, W: original shape
        patch_size: int
    Returns:
        x: Tensor of shape [B, T, C, H, W]
    """
    ph, pw = H // patch_size, W // patch_size
    patch_area = patch_size * patch_size
    total_patches = ph * pw
    patch_len = T * patch_area
 
    # Step 1: reverse indexing
    i_pix_idx = i_shape_flat_indices(patch_size, patch_size)
    reverse_pix_idx = torch.argsort(torch.tensor(i_pix_idx, device=out.device))
 

    # if(pw == 2):
    #     i_patch_idx = [(0,0), (1,0), (1,1), (0,1)]
    # elif(pw == 1):
    #     i_patch_idx = [(0,0)]
    # else:
    #     i_patch_idx = i_shape_indices(ph, pw, ph)
    i_patch_idx = s_shape_indices(ph, pw)
 
    # Step 2: split into per-patch chunks
    patch_tokens = out.split(patch_len, dim=-1)  # list of [B, C, T*P]
 
    # Step 3: fill full tensor with unpatchified data
    patches = torch.zeros(B, ph, pw, T, C, patch_area, device=out.device)
 
    for n, (ph_i, pw_i) in enumerate(i_patch_idx):
        patch = patch_tokens[n]  # [B, C, T*P]
        patch = rearrange(patch, 'b c (t p) -> b t c p', t=T)  # [B, T, C, P]
        patch = patch[..., reverse_pix_idx]  # undo i-shape flatten
        patches[:, ph_i, pw_i] = patch
 
    # Step 4: unpatchify to [B, T, C, H, W]
    x = rearrange(patches, 'b ph pw t c (h w) -> b t c (ph h) (pw w)', h=patch_size, w=patch_size)
    return x