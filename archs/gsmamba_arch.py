
import os
import warnings
import math
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from distutils.version import LooseVersion
from torch.nn.modules.utils import _pair, _single
import numpy as np
from einops.layers.torch import Rearrange

from functools import partial

from functools import reduce, lru_cache
from operator import mul
from einops import rearrange, repeat
from basicsr.archs.spynet_arch import SpyNet

import sys
sys.path.append('/hdd/laniko/MambaIR')
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from typing import Optional, Callable

from basicsr.archs.arch_util import s_shape_temporal_flatten, s_shape_temporal_unflatten, i_shape_temporal_flatten, i_shape_temporal_unflatten, i_shape_flat_indices, s_shape_flat_indices
from mamba_ssm import Mamba

import random
from basicsr.archs.implicit_alignment import ImplicitWarpModule
from basicsr.utils.registry import ARCH_REGISTRY

random.seed(10)   


class ModulatedDeformConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    # def forward(self, x, offset, mask):
    #     return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation,
    #                                  self.groups, self.deformable_groups)


class ModulatedDeformConvPack(ModulatedDeformConv):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConvPack, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            dilation=_pair(self.dilation),
            bias=True)
        self.init_weights()

    def init_weights(self):
        super(ModulatedDeformConvPack, self).init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    # def forward(self, x):
    #     out = self.conv_offset(x)
    #     o1, o2, mask = torch.chunk(out, 3, dim=1)
    #     offset = torch.cat((o1, o2), dim=1)
    #     mask = torch.sigmoid(mask)
    #     return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation,
    #                                  self.groups, self.deformable_groups)





class DCNv2PackFlowGuided(ModulatedDeformConvPack):
    """Flow-guided deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset residue. Default: 10.
        pa_frames (int): The number of parallel warping frames. Default: 2.

    Ref:
        BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment.

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(DCNv2PackFlowGuided, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d((1+1) * self.in_channels + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 3 * 9 * self.deformable_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        super(ModulatedDeformConvPack, self).init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset[-1].weight.data.zero_()
            self.conv_offset[-1].bias.data.zero_()

    def forward(self, x, x_flow_warpeds, x_current, flows):
        
        # import pdb; pdb.set_trace()
        out = self.conv_offset(torch.cat(x_flow_warpeds + [x_current] + flows, dim=1))
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset = offset + flows[0].flip(1).repeat(1, offset.size(1)//2, 1, 1)

        # mask
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding, self.dilation, mask)



def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True, use_pad_mask=False):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear' or 'nearest4'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.
        use_pad_mask (bool): only used for PWCNet, x is first padded with ones along the channel dimension.
            The mask is generated according to the grid_sample results of the padded dimension.


    Returns:
        Tensor: Warped image or feature map.
    """
    # assert x.size()[-2:] == flow.size()[1:3] # temporaily turned off for image-wise shift
    n, _, h, w = x.size()
    # create mesh grid
    # grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x)) # an illegal memory access on TITAN RTX + PyTorch1.9.1
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h, dtype=x.dtype, device=x.device), torch.arange(0, w, dtype=x.dtype, device=x.device))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow

    # if use_pad_mask: # for PWCNet
    #     x = F.pad(x, (0,0,0,0,0,1), mode='constant', value=1)

    # scale grid to [-1,1]
    if interp_mode == 'nearest4': # todo: bug, no gradient for flow model in this case!!! but the result is good
        vgrid_x_floor = 2.0 * torch.floor(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
        vgrid_x_ceil = 2.0 * torch.ceil(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
        vgrid_y_floor = 2.0 * torch.floor(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0
        vgrid_y_ceil = 2.0 * torch.ceil(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0

        output00 = F.grid_sample(x, torch.stack((vgrid_x_floor, vgrid_y_floor), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)
        output01 = F.grid_sample(x, torch.stack((vgrid_x_floor, vgrid_y_ceil), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)
        output10 = F.grid_sample(x, torch.stack((vgrid_x_ceil, vgrid_y_floor), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)
        output11 = F.grid_sample(x, torch.stack((vgrid_x_ceil, vgrid_y_ceil), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)

        return torch.cat([output00, output01, output10, output11], 1)

    else:
        vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

        # if use_pad_mask: # for PWCNet
        #     output = _flow_warp_masking(output)

        # TODO, what if align_corners=False
        return output



class Mlp_GEGLU(nn.Module):
    """ Multilayer perceptron with gated linear unit (GEGLU). Ref. "GLU Variants Improve Transformer".

    Args:
        x: (B, D, H, W, C)

    Returns:
        x: (B, D, H, W, C)
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc11 = nn.Linear(in_features, hidden_features)
        self.fc12 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.act(self.fc11(x)) * self.fc12(x)
        x = self.drop(x)
        x = self.fc2(x)

        return x


class WindowAttention(nn.Module):
    """ Window based multi-head mutual attention and self attention.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        mut_attn (bool): If True, add mutual attention to the module. Default: True
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # self attention with relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH
        self.register_buffer("relative_position_index", self.get_position_index(window_size))
        self.qkv_self = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        

        self.softmax = nn.Softmax(dim=-1)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """

        # self attention
        B_, N, C = x.shape
        qkv = self.qkv_self(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C
        x_out = self.attention(q, k, v, mask, (B_, N, C), relative_position_encoding=True)

        # projection
        x = self.proj(x_out)

        return x

    def attention(self, q, k, v, mask, x_shape, relative_position_encoding=True):
        B_, N, C = x_shape
        attn = (q * self.scale) @ k.transpose(-2, -1)

        if relative_position_encoding:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)  # Wd*Wh*Ww, Wd*Wh*Ww,nH
            attn = attn + relative_position_bias.permute(2, 0, 1).unsqueeze(0)  # B_, nH, N, N

        if mask is None:
            attn = self.softmax(attn)
        else:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask[:, :N, :N].unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        return x

    def get_position_index(self, window_size):
        ''' Get pair-wise relative position index for each token inside the window. '''

        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww

        return relative_position_index

    def get_sine_position_encoding(self, HW, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        """ Get sine position encoding """

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")

        if scale is None:
            scale = 2 * math.pi

        not_mask = torch.ones([1, HW[0], HW[1]])
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

        # BxCxHxW
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_embed = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos_embed.flatten(2).permute(0, 2, 1).contiguous()


class TMSA(nn.Module):
    """ Temporal Mutual Self Attention (TMSA).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for mutual and self attention.
        mut_attn (bool): If True, use mutual and self attention. Default: True.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_path (float, optional): Stochastic depth rate. Default: 0.0.
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(6, 8, 8),
                 shift_size=(0, 0, 0),
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False
                 ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpoint_attn = use_checkpoint_attn
        self.use_checkpoint_ffn = use_checkpoint_ffn

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias,
                                    qk_scale=qk_scale)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp_GEGLU(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1), mode='constant')

        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C

        # attention / shifted attention
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C

        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C

        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :]

        x = self.drop_path(x)

        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        # attention
        if self.use_checkpoint_attn:
            x = x + checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = x + self.forward_part1(x, mask_matrix)

        # feed-forward
        if self.use_checkpoint_ffn:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class TMSAG(nn.Module):
    """ Temporal Mutual Self Attention Group (TMSAG).

    Args:
        dim (int): Number of feature channels
        input_resolution (tuple[int]): Input resolution.
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (6,8,8).
        shift_size (tuple[int]): Shift size for mutual and self attention. Default: None.
        mut_attn (bool): If True, use mutual and self attention. Default: True.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=[6, 8, 8],
                 shift_size=None,
                 mlp_ratio=2.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = list(i // 2 for i in window_size) if shift_size is None else shift_size

        # build blocks
        self.blocks = nn.ModuleList([
            TMSA(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=[0, 0, 0] if i % 2 == 0 else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint_attn=use_checkpoint_attn,
                use_checkpoint_ffn=use_checkpoint_ffn
            )
            for i in range(depth)])

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for attention
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)

        for blk in self.blocks:
            x = blk(x, attn_mask)

        x = x.view(B, D, H, W, -1)
        x = rearrange(x, 'b d h w c -> b c d h w')

        return x



class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, nf, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().reshape(B, C, nf, H, W)
        x = self.dwconv(x)
        x = x.contiguous().flatten(2).transpose(1, 2)

        return x



class deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        return self.conv(x)


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


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)



class BasicModule(nn.Module):
    """Basic Module for SpyNet.
    """

    def __init__(self):
        super(BasicModule, self).__init__()

        self.basic_module = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)



def window_partition(x, window_size):
    """ Partition the input into windows. Attention will be conducted within the windows.

    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)

    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """ Reverse windows back to the original input. Attention was conducted within the windows.

    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)

    return x


def get_window_size(x_size, window_size, shift_size=None):
    """ Get the window size and the shift size """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    """ Compute attnetion mask for input of size (D, H, W). @lru_cache caches each stage results. """

    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


class Upsample(nn.Sequential):
    """Upsample module for video SR.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        assert LooseVersion(torch.__version__) >= LooseVersion('1.8.1'), \
            'PyTorch version >= 1.8.1 to support 5D PixelShuffle.'

        class Transpose_Dim12(nn.Module):
            """ Transpose Dim1 and Dim2 of a tensor."""

            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.transpose(1, 2)

        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv3d(num_feat, 4 * num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
                m.append(Transpose_Dim12())
                m.append(nn.PixelShuffle(2))
                m.append(Transpose_Dim12())
                m.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            m.append(nn.Conv3d(num_feat, num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
        elif scale == 3:
            m.append(nn.Conv3d(num_feat, 9 * num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
            m.append(Transpose_Dim12())
            m.append(nn.PixelShuffle(3))
            m.append(Transpose_Dim12())
            m.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            m.append(nn.Conv3d(num_feat, num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, nf, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, nf, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
        



class MambaBlock(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, mlp_ratio=4, drop=0., drop_path=0., act_layer=nn.GELU, num_heads=8, reverse=True, use_checkpoint_attn=True):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        # self.norm1 = RMSNorm(dim)
        
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v2",
                # use_fast_path=False,
        )
        self.norm2 = nn.LayerNorm(dim)
        # self.norm2 = RMSNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.reverse = reverse
        self.apply(self._init_weights)

        self.skip_scale= nn.Parameter(torch.ones(1, dim, 1, 1, 1))
        self.skip_scale2= nn.Parameter(torch.ones(1, 1, dim))
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def flip_even_groups(self, x, group_size=3):
        
        B, C, L = x.shape
        assert L % group_size == 0, "Length must be divisible by group size"
        
        num_groups = L // group_size
        x = x.view(B, C, num_groups, group_size)  # [B, C, G, group_size]

        # flip mask: [False, True, False, True, ...]
        flip_mask = torch.arange(num_groups, device=x.device) % 2 == 1  # [G]
        if flip_mask.sum() == 0:
            return x.view(B, C, L)

        # Select the even-indexed groups
        flipped = x[:, :, flip_mask].flip(-1)

        # Replace only the flipped ones
        x[:, :, flip_mask] = flipped

        return x.view(B, C, L)
    

    def forward(self, x, flows_backward, flows_forward, align_num, align_method=None):
        
        if self.reverse:
            x = x.transpose(-2,-1).contiguous()
            
        B, C, nf, H, W = x.shape
        assert C == self.dim
        
        
        center_idx = align_num
        x_orig = x.clone()
        
        x = self.get_center_aligned_features(x, flows_backward, flows_forward, center_idx, align_method)
        
        # import torchvision.utils as vutils
        # from torchvision.transforms.functional import to_pil_image
        # frames = x[0, :3].permute(1, 0, 2, 3)  # [8, 3, 64, 64]
        # frames = (frames - frames.min()) / (frames.max() - frames.min() + 1e-8)
        # grid = vutils.make_grid(frames, nrow=8, padding=2)
        # to_pil_image(grid).save("/hdd/laniko/AAAI/RethinkVSRAlignment/a_debug/x_aligned.png")

        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]

        
        x_flat = x.reshape(B, C, n_tokens)
        x_flat = x_flat.reshape(B, C, nf, H, W)
        
    
        ## temporal first ! ##
        x_flat = x_flat.permute(0,1,3,4,2).contiguous().reshape(B, C, n_tokens)
        ######################
        
        x_flat = x_flat.transpose(-2,-1).contiguous()
        
        
        # Bi-Mamba layer
        mamba_output = self.mamba(self.norm1(x_flat)).transpose(-2,-1).contiguous()
        
        mamba_output = mamba_output.reshape(B,C,H,W,nf).permute(0,1,4,2,3).contiguous()
        x_mamba = self.recover_original_features(mamba_output, flows_backward, flows_forward, center_idx, x_orig, align_method)

        x_mamba = self.skip_scale * x_orig + x_mamba
        x_mamba = x_mamba.reshape(B, C, -1).transpose(-2,-1).contiguous()
        
        x_mamba = self.skip_scale2 * x_mamba + self.mlp(self.norm2(x_mamba), nf, H, W)
        out = x_mamba.transpose(-1, -2).reshape(B, C, nf, H, W)
        
        
        if self.reverse:
            out = out.transpose(-2,-1).contiguous()
      
        return out




    def get_center_aligned_features(self, x, flows_forward, flows_backward, center_idx, align_method=None):
        """
        Align all frames to the center frame using composed flow and implicit warping.

        Args:
            x: Tensor of shape (B, C, T, H, W)
            flows_forward: list with 1 tensor, shape (B, T-1, 2, H, W) — i+1 → i
            flows_backward: list with 1 tensor, shape (B, T-1, 2, H, W) — i → i+1
            center_idx: int, index of center frame (e.g., 2)

        Returns:
            aligned_feats: (B, C, T, H, W) — All features aligned to center frame
        """
        B, C, T, H, W = x.shape
        x_center = x[:, :, center_idx, :, :]
        aligned = []

        # flow aliases
        flow_fw = flows_backward  # i → i+1
        flow_bw = flows_forward  # i+1 → i

        if self.reverse:
            flow_fw = flow_fw.transpose(-1, -2).contiguous()
            flow_x = flow_fw[:, :, 0].clone()
            flow_y = flow_fw[:, :, 1].clone()
            flow_fw[:, :, 0] = flow_y
            flow_fw[:, :, 1] = flow_x

            flow_bw = flow_bw.transpose(-1, -2).contiguous()
            flow_x = flow_bw[:, :, 0].clone()
            flow_y = flow_bw[:, :, 1].clone()
            flow_bw[:, :, 0] = flow_y
            flow_bw[:, :, 1] = flow_x
            

        for i in range(T):
            if i == center_idx:
                zero_flow = torch.zeros_like(flow_fw[:, 0])  # (B, 2, H, W)
                x_warped = align_method(x[:, :, i, :, :], x_center, zero_flow.permute(0, 2, 3, 1))
                
                # x_warped = x[:, :, i, :, :]
                aligned.append(x_warped)
            elif i < center_idx:
                # Forward chain: i → center
                flow = flow_fw[:, i, :, :, :]

                for j in range(i + 1, center_idx):
                    flow = flow + flow_warp(flow_fw[:, j, :, :, :], flow.permute(0, 2, 3, 1))


                x_warped = align_method(x[:, :, i, :, :], x_center, flow.permute(0, 2, 3, 1))
                
                # x_warped = flow_warp(x[:, :, i, :, :], flow.permute(0, 2, 3, 1))
                # x_warped = align_method(x[:, :, i, :, :], [x_warped], x_center, [flow])
                
                aligned.append(x_warped)
            else:
                # Backward chain: i → center
                flow = flow_bw[:, i - 1, :, :, :]

                for j in range(i - 2, center_idx - 1, -1):
                    flow = flow + flow_warp(flow_bw[:, j, :, :, :], flow.permute(0, 2, 3, 1))

                x_warped = align_method(x[:, :, i, :, :], x_center, flow.permute(0, 2, 3, 1))
                
                # x_warped = flow_warp(x[:, :, i, :, :], flow.permute(0, 2, 3, 1))
                # x_warped = align_method(x[:, :, i, :, :], [x_warped], x_center, [flow])
                
                aligned.append(x_warped)

        # Stack along time dimension
        aligned_feats = torch.stack(aligned, dim=2)  # (B, C, T, H, W)
        
    
        
        return aligned_feats





    def recover_original_features(self, x_aligned, flows_forward, flows_backward, center_idx, original_x, align_method=None):
        """
        Warp center-aligned features back to original positions.

        Args:
            x_aligned: (B, C, T, H, W), aligned to center
            flows_forward: list with 1 tensor (B, T-1, 2, H, W), i+1 → i
            flows_backward: list with 1 tensor (B, T-1, 2, H, W), i → i+1
            center_idx: int

        Returns:
            x_recovered: (B, C, T, H, W)
        """
        B, C, T, H, W = x_aligned.shape
        x_center = x_aligned[:, :, center_idx, :, :]
        recovered = []
        x_origin = original_x

        flow_fw = flows_backward  # i → i+1
        flow_bw = flows_forward  # i+1 → i


        if self.reverse:
            flow_fw = flow_fw.transpose(-1, -2).contiguous()
            flow_x = flow_fw[:, :, 0].clone()
            flow_y = flow_fw[:, :, 1].clone()
            flow_fw[:, :, 0] = flow_y
            flow_fw[:, :, 1] = flow_x

            flow_bw = flow_bw.transpose(-1, -2).contiguous()
            flow_x = flow_bw[:, :, 0].clone()
            flow_y = flow_bw[:, :, 1].clone()
            flow_bw[:, :, 0] = flow_y
            flow_bw[:, :, 1] = flow_x

        for i in range(T):
            if i == center_idx:
                # zero_flow = torch.zeros_like(flow_fw[:, 0])  # (B, 2, H, W)
                # x_warped = align_method(x_center, x_origin[:, :, i, :, :], zero_flow.permute(0, 2, 3, 1))
                x_warped = x_center
                
                recovered.append(x_warped)
            elif i < center_idx:
                # Backward: center → i (follow flow_bw backward)
                flow = flow_bw[:, center_idx - 1, :, :, :]

                for j in range(center_idx - 2, i - 1, -1):
                    flow = flow + flow_warp(flow_bw[:, j, :, :, :], flow.permute(0, 2, 3, 1))

                # x_warped = align_method(x_aligned[:, :, i, ...], x_origin[:, :, i, :, :], flow.permute(0, 2, 3, 1))
                x_warped = flow_warp(x_aligned[:, :, i, ...], flow.permute(0, 2, 3, 1))
                
            
                recovered.append(x_warped)
            else:
                # Forward: center → i (follow flow_fw forward)
                flow = flow_fw[:, center_idx, :, :, :]
                for j in range(center_idx + 1, i):
                    flow = flow + flow_warp(flow_fw[:, j, :, :, :], flow.permute(0, 2, 3, 1))

                # x_warped = align_method(x_aligned[:, :, i, ...], x_origin[:, :, i, :, :], flow.permute(0, 2, 3, 1))
                x_warped = flow_warp(x_aligned[:, :, i, ...], flow.permute(0, 2, 3, 1))
        
                recovered.append(x_warped)

        return torch.stack(recovered, dim=2)



    def compute_occlusion_mask(self, flow_fw, flow_bw, gamma1=0.01, gamma2=0.5, threshold=0.5):
        """
        Forward & Backward Consistency를 기반으로 Confidence Map을 계산하여 Occlusion Mask 생성
        
        Args:
            flow_fw: (B, T, 2, H, W)  -> Forward Optical Flow (t → t+1)
            flow_bw: (B, T, 2, H, W)  -> Backward Optical Flow (t+1 → t)
            gamma1: float  -> 정규화 상수 (기본값 0.01)
            gamma2: float  -> 정규화 상수 (기본값 0.5)
            threshold: float  -> Confidence Map 임계값 (Occlusion Mask를 위한 기준)

        Returns:
            confidence_map: (B, T, H, W)  -> Forward-Backward Consistency 기반 Confidence Map
            occlusion_mask: (B, T, H, W)  -> Confidence 값이 낮을수록 Occlusion으로 판별한 Mask (1 = Occluded)
        """
        B, T, _, H, W = flow_fw.shape  # (Batch, Time, 2, Height, Width)

        # 좌표 grid 생성
        x = torch.linspace(0, W-1, W).repeat(H, 1).to(flow_fw.device)
        y = torch.linspace(0, H-1, H).repeat(W, 1).T.to(flow_fw.device)
        grid = torch.stack((x, y), dim=0).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1, 1)  # (B, T, 2, H, W)

        # Forward Flow 적용한 위치
        warped_grid = grid + flow_fw  # (B, T, 2, H, W)

        # 좌표 정규화 (0~W-1 → -1~1)
        warped_x = (warped_grid[:, :, 0] / (W - 1)) * 2 - 1
        warped_y = (warped_grid[:, :, 1] / (H - 1)) * 2 - 1
        warped_grid_for_sampling = torch.stack((warped_x, warped_y), dim=-1).view(B * T, H, W, 2)

        # Grid Sample을 위한 차원 변환 (B, T, 2, H, W) -> (B*T, 2, H, W)
        flow_bw_reshaped = flow_bw.view(B * T, 2, H, W)

        # Bilinear interpolation으로 Backward Flow 샘플링
        sampled_flow_bw = F.grid_sample(flow_bw_reshaped, warped_grid_for_sampling, align_corners=True, mode='bilinear')
        sampled_flow_bw = sampled_flow_bw.view(B, T, 2, H, W)

        # Forward + Backward Flow Norm Squared
        flow_diff = (flow_fw + sampled_flow_bw).norm(dim=2) ** 2  # (B, T, H, W)

        # Normalization Term
        norm_term = gamma1 * ((flow_fw.norm(dim=2) ** 2) + (sampled_flow_bw.norm(dim=2) ** 2)) + gamma2  # (B, T, H, W)

        # Confidence Map 계산
        confidence_map = torch.exp(-flow_diff / norm_term)  # (B, T, H, W)

        # Occlusion Mask 생성 (Confidence가 낮은 영역을 Occlusion으로 간주)
        occlusion_mask = (confidence_map < threshold).float()  # (B, T, H, W)

        return confidence_map, occlusion_mask

class Stage(nn.Module):
    """Residual Temporal Mutual Self Attention Group and Parallel Warping.

    Args:
        in_dim (int): Number of input channels.
        dim (int): Number of channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        pa_frames (float): Number of warpped frames. Default: 2.
        deformable_groups (float): Number of deformable groups. Default: 16.
        reshape (str): Downscale (down), upscale (up) or keep the size (none).
        max_residue_magnitude (float): Maximum magnitude of the residual of optical flow.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(self,
                 in_dim,
                 dim,
                 num_heads,
                 norm_layer=nn.LayerNorm,
                 attn_depth=2,
                 pa_frame=2,
                 use_checkpoint_attn=True,
                 use_checkpoint_ffn=True,
                 ):
        super(Stage, self).__init__()

        self.attn_blocks = TMSAG(dim=dim, num_heads=num_heads, window_size=(2,8,8), shift_size=None,
                mlp_ratio=2, norm_layer=norm_layer, use_checkpoint_attn=use_checkpoint_attn,
                use_checkpoint_ffn=use_checkpoint_ffn, depth=attn_depth)

        self.mamba_block = MambaBlock(dim=dim, d_state=16, d_conv=4, expand=2, mlp_ratio=4, act_layer=nn.GELU, num_heads=num_heads, reverse=False, use_checkpoint_attn=use_checkpoint_attn)

        self.pa_frame = pa_frame

    def forward_attn_blocks_chunked(self, x, chunk_size=24, stride=12):
        """Split long sequence temporally and process with attention block chunk by chunk."""
        B, C, T, H, W = x.shape
        output = torch.zeros_like(x)
        count_map = torch.zeros_like(x)

        for start in range(0, T, stride):
            end = min(start + chunk_size, T)
            chunk = x[:, :, start:end]  # [B, C, t, H, W]

            # attention 적용
            out_chunk = self.attn_blocks(chunk)

            # output에 합치기
            output[:, :, start:end] += out_chunk
            count_map[:, :, start:end] += 1

            if end == T:
                break

        # 평균을 내서 stitching
        output = output / torch.clamp(count_map, min=1.0)
        return output


    def forward(self, x, flows_backward, flows_forward, align_method=None):
        shortcut = x
        
        # if x.shape[2] > 50:
        #     x = self.forward_attn_blocks_chunked(x, chunk_size=20, stride=2)
        # else:
        #     x = self.attn_blocks(x)
    
        
        x = self.attn_blocks(x)
        
        nf = x.shape[2]  # 16
        clip_radius = self.pa_frame
        zero_flow = torch.zeros_like(flows_backward[:, 0, ...])
        
        for idx in range(nf):
            curr_clip = []
            flows_bw = []
            flows_fw = []

            for i in range(-clip_radius, clip_radius + 1):
                target_idx = idx + i

                if target_idx < 0:
                    final_idx = 0
                    flows_bw.append(zero_flow)
                    flows_fw.append(zero_flow)
                    
                elif target_idx >= nf:
                    final_idx = nf - 1
                    flows_bw.append(zero_flow)
                    flows_fw.append(zero_flow)
                else:
                    final_idx = target_idx
                    if target_idx == nf - 1:
                        flows_bw.append(zero_flow)
                        flows_fw.append(zero_flow)
                        
                    if(final_idx < nf - 1):
                        flows_bw.append(flows_backward[:, final_idx, ...])
                        flows_fw.append(flows_forward[:, final_idx, ...])


                curr_clip.append(x[:, :, final_idx, ...])
                

            curr_x = torch.stack(curr_clip, dim=2)
            flows_bw = torch.stack(flows_bw[:-1], dim=1)
            flows_fw = torch.stack(flows_fw[:-1], dim=1)
            
            prop_x = self.mamba_block(curr_x, flows_bw, flows_fw, clip_radius, align_method=align_method)
            
            # import pdb; pdb.set_trace()
            
            for i in range(-clip_radius, clip_radius + 1):
                target_idx = idx + i
                if target_idx < 0:
                    continue
                elif target_idx >= nf:
                    continue
                else:
                    x[:, :, target_idx] = prop_x[:, :, i+clip_radius]
        
        x = shortcut + x

        return x


    # def forward(self, x, flows_backward, flows_forward, align_method=None):
    #     shortcut = x

    #     print(f"[DEBUG] Input: x mean={x.mean().item():.4f}, std={x.std().item():.4f}")


    #     # x = self.test_norm(x.transpose(1,4).contiguous()).transpose(1,4).contiguous()
        
        
    #     # Attention block
    #     x = self.attn_blocks(x)
        
    #     # x = self.test_norm(x.transpose(1,4).contiguous()).transpose(1,4).contiguous()
        
    #     print(f"[DEBUG] After attn_blocks: mean={x.mean().item():.4f}, std={x.std().item():.4f}")
        
        
    #     # x = self.test_norm(x.transpose(1,4).contiguous()).transpose(1,4).contiguous()
        
        
        
    #     nf = x.shape[2]
    #     clip_radius = self.pa_frame
    #     zero_flow = torch.zeros_like(flows_backward[:, 0, ...])

    #     for idx in range(nf):
    #         curr_clip = []
    #         flows_bw = []
    #         flows_fw = []

    #         for i in range(-clip_radius, clip_radius + 1):
    #             target_idx = idx + i

    #             if target_idx < 0:
    #                 final_idx = 0
    #                 flows_bw.append(zero_flow)
    #                 flows_fw.append(zero_flow)
    #             elif target_idx >= nf:
    #                 final_idx = nf - 1
    #                 flows_bw.append(zero_flow)
    #                 flows_fw.append(zero_flow)
    #             else:
    #                 final_idx = target_idx
    #                 if target_idx == nf - 1:
    #                     flows_bw.append(zero_flow)
    #                     flows_fw.append(zero_flow)
    #                 if final_idx < nf - 1:
    #                     flows_bw.append(flows_backward[:, final_idx, ...])
    #                     flows_fw.append(flows_forward[:, final_idx, ...])

    #             curr_clip.append(x[:, :, final_idx, ...])

    #         curr_x = torch.stack(curr_clip, dim=2)
    #         flows_bw = torch.stack(flows_bw[:-1], dim=1)
    #         flows_fw = torch.stack(flows_fw[:-1], dim=1)

    #         # print(f"[DEBUG] Mamba input at frame {idx}: mean={curr_x.mean().item():.4f}, std={curr_x.std().item():.4f}")
    #         prop_x = self.mamba_block(curr_x, flows_bw, flows_fw, clip_radius, align_method=align_method)
    #         # print(f"[DEBUG] Mamba output at frame {idx}: mean={prop_x.mean().item():.4f}, std={prop_x.std().item():.4f}")

    #         for i in range(-clip_radius, clip_radius + 1):
    #             target_idx = idx + i
    #             if 0 <= target_idx < nf:
    #                 x[:, :, target_idx] = prop_x[:, :, i + clip_radius]

    #     print(f"[DEBUG] After mamba prop: mean={x.mean().item():.4f}, std={x.std().item():.4f}")

    #     x = shortcut + x
    #     print(f"[DEBUG] After residual add: mean={x.mean().item():.4f}, std={x.std().item():.4f}")
        
    #     import pdb; pdb.set_trace()

    #     return x



class PropLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 dim,
                 num_heads,
                 norm_layer=nn.LayerNorm,
                 attn_depth=4,
                 pa_frame=2,
                 is_even=False,
                 use_checkpoint_attn=True,
                 use_checkpoint_ffn=True,
                 ):
        super(PropLayer, self).__init__()
        
        self.is_even = is_even
        
        self.stage = Stage(in_dim=in_dim,
                      dim=dim,
                      attn_depth=attn_depth,
                      num_heads=num_heads,
                      pa_frame=pa_frame,
                      norm_layer=norm_layer,
                      use_checkpoint_attn=use_checkpoint_attn,
                      use_checkpoint_ffn=use_checkpoint_ffn,
                      )
        
    def forward(self, x, flows_backward, flows_forward, align_method=None):

        if(self.is_even):
            flows_temp = flows_backward.clone()
            flows_backward = torch.flip(flows_forward.clone(), dims=[1])
            flows_forward = torch.flip(flows_temp, dims=[1])
        
        x = self.stage(x, flows_backward, flows_forward, align_method=align_method)
        
        x = torch.flip(x, dims=[2])

        
        return x
        
        
        
@ARCH_REGISTRY.register()
class GSMamba(nn.Module):
    """ GSMamba: flow-guided implicit-warp alignment + bidirectional Mamba + windowed attention for video super-resolution.
        A PyTorch impl of : `VRT: A Video Restoration Transformer`  -
          https://arxiv.org/pdf/2201.00000

    Args:
        upscale (int): Upscaling factor. Set as 1 for video deblurring, etc. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        out_chans (int): Number of output image channels. Default: 3.
        img_size (int | tuple(int)): Size of input image. Default: [6, 64, 64].
        window_size (int | tuple(int)): Window size. Default: (6,8,8).
        depths (list[int]): Depths of each Transformer stage.
        indep_reconsts (list[int]): Layers that extract features of different frames independently.
        embed_dims (list[int]): Number of linear projection output channels.
        num_heads (list[int]): Number of attention head of each stage.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (obj): Normalization layer. Default: nn.LayerNorm.
        spynet_path (str): Pretrained SpyNet model path.
        pa_frames (float): Number of warpped frames. Default: 2.
        deformable_groups (float): Number of deformable groups. Default: 16.
        recal_all_flows (bool): If True, derive (t,t+2) and (t,t+3) flows from (t,t+1). Default: False.
        nonblind_denoising (bool): If True, conduct experiments on non-blind denoising. Default: False.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
        no_checkpoint_attn_blocks (list[int]): Layers without torch.checkpoint for attention modules.
        no_checkpoint_ffn_blocks (list[int]): Layers without torch.checkpoint for feed-forward modules.
    """

    def __init__(self,
                 upscale=4,
                 in_chans=3,
                 out_chans=3,
                 prop_depth=4,
                 attn_depth=2,
                 pa_frame=2,
                 embed_dims=144,
                 num_heads=8,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint_attn=True,
                 use_checkpoint_ffn=True,
                 is_train=True,
                 spynet_path='experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth',
                 ):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.upscale = upscale
         
        self.is_train = is_train
            
        self.conv_first = nn.Conv3d(in_chans, embed_dims, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.prop_layers = nn.ModuleList()
        
        
        self.implicit_align = ImplicitWarpModule(      
            dim=embed_dims,
            pe_dim=embed_dims,
            num_heads=num_heads,
            pe_temp=0.01,
            use_checkpoint_attn=use_checkpoint_attn,)

        # self.implicit_align = ImplicitWarpModule(      
        #     dim=embed_dims,
        #     pe_dim=embed_dims,
        #     num_heads=num_heads,
        #     pe_temp=0.01,
        #     use_checkpoint_attn=False,)
        

        
        for i in range(prop_depth):
            self.prop_layers.append(
                PropLayer(in_dim=embed_dims,
                      dim=embed_dims,
                      attn_depth=attn_depth,
                      num_heads=num_heads,
                      pa_frame=pa_frame,
                      norm_layer=norm_layer,
                      is_even=False if i%2==0 else True,
                      use_checkpoint_attn=use_checkpoint_attn,
                      use_checkpoint_ffn=use_checkpoint_ffn,
                      )
            )
            
            

        # reconstruction
        self.norm = norm_layer(embed_dims)
        self.conv_after_body = nn.Linear(embed_dims, embed_dims)

        num_feat = 64
        self.conv_before_upsample = nn.Sequential(
            nn.Conv3d(embed_dims, num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(inplace=True))

        
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv3d(num_feat, out_chans, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        
        self.spynet = SpyNet(spynet_path)
            
        

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

    def reflection_pad2d(self, x, pad=1):
        """ Reflection padding for any dtypes (torch.bfloat16.

        Args:
            x: (tensor): BxCxHxW
            pad: (int): Default: 1.
        """

        x = torch.cat([torch.flip(x[:, :, 1:pad+1, :], [2]), x, torch.flip(x[:, :, -pad-1:-1, :], [2])], 2)
        x = torch.cat([torch.flip(x[:, :, :, 1:pad+1], [3]), x, torch.flip(x[:, :, :, -pad-1:-1], [3])], 3)
        return x


    
    def forward(self, x):

        B,T,C,H,W = x.shape
        
        x_lq = x.clone()
        
        flows_backward, flows_forward = self.get_flows(x)   # 4 * [1, 6, 2, 64, 64]

        

        x = self.conv_first(x.transpose(1, 2))
        x = x + self.conv_after_body(
            self.forward_features(x, flows_backward, flows_forward).transpose(1, 4)).transpose(1, 4)
        
        x = self.conv_last(self.upsample(self.conv_before_upsample(x))).transpose(1, 2)
        
        _, _, C, H, W = x.shape
        return x + torch.nn.functional.interpolate(x_lq, size=(C, H, W), mode='trilinear', align_corners=False)



    def get_flows(self, x):
        ''' Get flows for 2 frames, 4 frames or 6 frames.'''
        flows_backward, flows_forward = self.get_flow_2frames(x)

        return flows_backward, flows_forward

    def get_flow_2frames(self, x):
        '''Get flow between frames t and t+1 from x.'''

        b, n, c, h, w = x.size()
        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        # backward
        flows_backward = self.spynet(x_1, x_2)
        flows_backward = flows_backward.reshape(b, n-1, 2, h, w)

        # forward
        flows_forward = self.spynet(x_2, x_1)
        flows_forward = flows_forward.reshape(b, n-1, 2, h, w)

        return flows_backward, flows_forward
    

    def forward_features(self, x, flows_backward, flows_forward):
        '''Main network for feature extraction.'''
        
        for layer in self.prop_layers:
            x = layer(x, flows_backward, flows_forward, align_method=self.implicit_align)

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')

        return x