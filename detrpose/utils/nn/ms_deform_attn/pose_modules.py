import math
import functools
from typing import List

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from detrpose.utils.nn.ms_deform_attn.functions import (
    deformable_attention_core_func,
    deformable_attention_core_func_v2,
    pose_deformable_attention_core_func,
)


class PoseMSDeformableAttention(nn.Module):
    def __init__(
            self,
            embed_dim=256,
            num_levels=4,
            num_heads=8,
            num_points=4,
            use_4D_normalizer=False
    ):
        """
        Multi-Scale Deformable Attention Module
        :param embed_dim      hidden dimension
        :param num_levels     number of feature levels
        :param num_heads      number of attention heads
        :param num_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                'embed_dim must be divisible by num_heads, but got {} and {}'.format(embed_dim, num_heads)
            )
        _d_per_head = embed_dim // num_heads

        self.embed_dim = embed_dim
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)

        self.use_4D_normalizer = use_4D_normalizer

        self._reset_parameters()

    def _reset_parameters(self):
        init.constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
                grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        ).view(self.num_heads, 1, 1, 2).repeat(
            1,
            self.num_levels,
            self.num_points,
            1
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i % 4 + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        if self.num_points % 4 != 0:
            init.constant_(self.sampling_offsets.bias, 0.)
        init.constant_(self.attention_weights.weight.data, 0.)
        init.constant_(self.attention_weights.bias.data, 0.)

    def forward(self, query, reference_points, value, input_spatial_shapes):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, num_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, num_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (num_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (num_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        # assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.num_heads, self.num_levels, self.num_points, 2
        )

        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Len_q, self.num_heads, self.num_levels, self.num_points
        )

        # N, Len_q, num_heads, num_levels, num_points, 2
        reference_points = torch.transpose(reference_points, 2, 3).flatten(1, 2)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(input_spatial_shapes, device=query.device)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            if self.use_4D_normalizer:
                offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
                sampling_locations = reference_points[:, :, None, :, None, :2] \
                                     + sampling_offsets / offset_normalizer[None, None, None, :, None, :] * \
                                     reference_points[:, :, None, :, None, 2:] * 0.5
            else:
                sampling_locations = reference_points[:, :, None, :, None, :2] \
                                     + sampling_offsets / self.num_points * reference_points[
                                         :, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

        output = pose_deformable_attention_core_func(
            value, input_spatial_shapes, sampling_locations, attention_weights)
        return output

