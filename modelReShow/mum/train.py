#!/usr/bin/env python
# coding: utf-8

# In[2]:


import math
import time
import os
import glob
import random
import warnings
from collections import deque
from datetime import datetime
from typing import Callable, Tuple, Union, Optional, Literal, List, Any
from functools import partial
import numpy as np
import wandb
from PIL import Image
from torchvision import transforms
import torch 
from torch.optim import AdamW
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# DDP 多卡训练
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext


# In[3]:


# 图片大小为元组
def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)

class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        x = self.proj(x)  # B C H W 卷积将每张图片变成一个有embed_dim维度的向量,每个向量的形状为 (B, C, H/patch_H, W/patch_W)
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C flatten(2)将H和W维度展平，变成(B, C, HW)，然后transpose(1, 2)交换C和HW维度，变成(B, HW, C)
        x = self.norm(x) # 归一化
        if not self.flatten_embedding: #
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    # 计算量统计
    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

    # 重置参数
    def reset_parameters(self):
        k = 1 / (self.in_chans * (self.patch_size[0] ** 2))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))


# In[4]:


# rope位置编码，不用参数，x、y分离计算
class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: Optional[float] = 100.0,
        min_period: Optional[float] = None,
        max_period: Optional[float] = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: Optional[float] = None,
        jitter_coords: Optional[float] = None,
        rescale_coords: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("必须提供 `base` 或 `min_period`+`max_period` 之一。")

        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        # 需要 persistent=True，因为我们用 teacher.load_state_dict(student.state_dict()) 来初始化 teacher
        self.dtype = dtype  # 不要依赖 self.periods.dtype
        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        # 准备坐标，范围 [-1, +1]
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / max_HW  # [W]
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / min_HW  # [W]
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H  # [H]
            coords_w = torch.arange(0.5, W, **dd) / W  # [W]
        else:
            raise ValueError(f"未知的 normalize_coords: {self.normalize_coords}")
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0  # 将范围从 [0, 1] 移到 [-1, +1]

        # 平移坐标：在 [-shift, shift] 范围内添加均匀随机值
        if self.training and self.shift_coords is not None:
            shift_hw = torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords += shift_hw[None, :]

        # 抖动坐标：将范围 [-1, 1] 乘以 [1/jitter, jitter] 范围内的对数均匀值
        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_hw = torch.empty(2, **dd).uniform_(jitter_min, jitter_max).exp()
            coords *= jitter_hw[None, :]

        # 缩放坐标：将范围 [-1, 1] 乘以 [1/rescale, rescale] 范围内的对数均匀值
        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_hw = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords *= rescale_hw

        # 计算角度和 sin/cos
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]  # [HW, 2, D//4]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        angles = angles.tile(2)  # [HW, D]
        cos = torch.cos(angles)  # [HW, D]
        sin = torch.sin(angles)  # [HW, D]

        return (sin, cos)  # 2 * [HW, D]

    # 初始化权重
    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (
                2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2)
            )  # [D//4]
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head // 4, device=device, dtype=dtype)  # [D//4] 范围 [0, 1]
            periods = base**exponents  # 范围 [1, max_period / min_period]
            periods = periods / base  # 范围 [min_period / max_period, 1]
            periods = periods * self.max_period  # 范围 [min_period, max_period]
        self.periods.data = periods


# In[5]:


class LinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.full_like(self.bias, fill_value=math.nan))

    def forward(self, input: Tensor) -> Tensor:
        masked_bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(input, self.weight, masked_bias)


# In[6]:


# RoPE-related functions:
def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)

def cat_keep_shapes(x_list: List[Tensor]) -> Tuple[Tensor, List[Tuple[int]], List[int]]:
    shapes = [x.shape for x in x_list]
    num_tokens = [x.select(dim=-1, index=0).numel() for x in x_list]
    flattened = torch.cat([x.flatten(0, -2) for x in x_list])
    return flattened, shapes, num_tokens

def uncat_with_shapes(flattened: Tensor, shapes: List[Tuple[int]], num_tokens: List[int]) -> List[Tensor]:
    outputs_splitted = torch.split_with_sizes(flattened, num_tokens, dim=0)
    shapes_adjusted = [shape[:-1] + torch.Size([flattened.shape[-1]]) for shape in shapes]
    outputs_reshaped = [o.reshape(shape) for o, shape in zip(outputs_splitted, shapes_adjusted)]
    return outputs_reshaped
def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
def named_apply(
    fn: Callable,
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


# In[7]:


class ListForwardMixin(object):
    def forward(self, x: Tensor):
        raise NotImplementedError

    def forward_list(self, x_list: List[Tensor]) -> List[Tensor]:
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        x_flat = self.forward(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)


class Mlp(nn.Module, ListForwardMixin):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        device=None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, device=device)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, device=device)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# In[8]:


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(torch.empty(dim, device=device))
        self.init_values = init_values

    def reset_parameters(self):
        nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


# In[9]:


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_rope(self, q: Tensor, k: Tensor, rope: Union[Tensor, Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]

        q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def forward(self, x: Tensor, attn_bias=None, rope: Tensor = None) -> Tensor:
        qkv = self.qkv(x)
        attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        return x

    def forward_list(self, x_list, attn_bias=None, rope_list=None) -> List[Tensor]:
        assert len(x_list) == len(rope_list)  # should be enforced by the Block
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
        att_out = []
        for _, (qkv, _, rope) in enumerate(zip(qkv_list, shapes, rope_list)):
            att_out.append(self.compute_attention(qkv, attn_bias=attn_bias, rope=rope))
        x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
        x_flat = self.proj(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)

    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        # self._last_q = q.detach().cpu()
        # self._last_k = k.detach().cpu()
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2)
        return x.reshape([B, N, C])


# In[10]:


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        mask_k_bias: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.sample_drop_ratio = drop_path

    @staticmethod
    def _maybe_index_rope(rope: Optional[tuple[Tensor, Tensor]], indices: Tensor) -> Optional[tuple[Tensor, Tensor]]:
        if rope is None:
            return None

        sin, cos = rope
        assert sin.ndim == cos.ndim
        if sin.ndim == 4:
            # If the rope embedding has a batch dimension (is different for each batch element), index into it
            return sin[indices], cos[indices]  # [batch, heads, patches, embed_dim]
        else:
            # No batch dimension, do not index
            return sin, cos  # [heads, patches, embed_dim] or [patches, embed_dim]

    def _forward(self, x: Tensor, rope=None) -> Tensor:
        """
        This is the reference implementation for a single tensor, matching what is done below for a list.
        We call the list op on [x] instead of this function.
        """
        b, _, _ = x.shape
        sample_subset_size = max(int(b * (1 - self.sample_drop_ratio)), 1)
        residual_scale_factor = b / sample_subset_size

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_1 = x[indices_1]
            rope_subset = self._maybe_index_rope(rope, indices_1)
            residual_1 = self.attn(self.norm1(x_subset_1), rope=rope_subset)

            x_attn = torch.index_add(
                x,
                dim=0,
                source=self.ls1(residual_1),
                index=indices_1,
                alpha=residual_scale_factor,
            )

            indices_2 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_2 = x_attn[indices_2]
            residual_2 = self.mlp(self.norm2(x_subset_2))

            x_ffn = torch.index_add(
                x_attn,
                dim=0,
                source=self.ls2(residual_2),
                index=indices_2,
                alpha=residual_scale_factor,
            )
        else:
            x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
            x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))

        return x_ffn

    def _forward_list(self, x_list: List[Tensor], rope_list=None) -> List[Tensor]:
        """
        This list operator concatenates the tokens from the list of inputs together to save
        on the elementwise operations. Torch-compile memory-planning allows hiding the overhead
        related to concat ops.
        """
        b_list = [x.shape[0] for x in x_list]
        sample_subset_sizes = [max(int(b * (1 - self.sample_drop_ratio)), 1) for b in b_list]
        residual_scale_factors = [b / sample_subset_size for b, sample_subset_size in zip(b_list, sample_subset_sizes)]

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_1_list = [x[indices_1] for x, indices_1 in zip(x_list, indices_1_list)]

            if rope_list is not None:
                rope_subset_list = [
                    self._maybe_index_rope(rope, indices_1) for rope, indices_1 in zip(rope_list, indices_1_list)
                ]
            else:
                rope_subset_list = rope_list

            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_1_list)
            norm1 = uncat_with_shapes(self.norm1(flattened), shapes, num_tokens)
            residual_1_list = self.attn.forward_list(norm1, rope_list=rope_subset_list)

            residual_1_list = [r.to(dtype=x_list[0].dtype) for r in residual_1_list]

            x_attn_list = [
                torch.index_add(
                    x,
                    dim=0,
                    source=self.ls1(residual_1),
                    index=indices_1,
                    alpha=residual_scale_factor,
                )
                for x, residual_1, indices_1, residual_scale_factor in zip(
                    x_list, residual_1_list, indices_1_list, residual_scale_factors
                )
            ]

            indices_2_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_2_list = [x[indices_2] for x, indices_2 in zip(x_attn_list, indices_2_list)]
            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_2_list)
            norm2_flat = self.norm2(flattened)
            norm2_list = uncat_with_shapes(norm2_flat, shapes, num_tokens)

            residual_2_list = self.mlp.forward_list(norm2_list)

            residual_2_list = [r.to(dtype=x_attn_list[0].dtype) for r in residual_2_list]

            x_ffn = [
                torch.index_add(
                    x_attn,
                    dim=0,
                    source=self.ls2(residual_2),
                    index=indices_2,
                    alpha=residual_scale_factor,
                )
                for x_attn, residual_2, indices_2, residual_scale_factor in zip(
                    x_attn_list, residual_2_list, indices_2_list, residual_scale_factors
                )
            ]
        else:
            x_out = []
            for x, rope in zip(x_list, rope_list):
                x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
                x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
                x_out.append(x_ffn)
            x_ffn = x_out

        return x_ffn

    def forward(self, x_or_x_list, rope_or_rope_list=None) -> List[Tensor]:
        if isinstance(x_or_x_list, Tensor):
            # for reference:
            # return self._forward(x_or_x_list, rope=rope_or_rope_list)
            # in order to match implementations we call the list op:
            return self._forward_list([x_or_x_list], rope_list=[rope_or_rope_list])[0]
        elif isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for x in x_or_x_list]
            # return [self._forward(x, rope=rope) for x, rope in zip(x_or_x_list, rope_or_rope_list)]
            return self._forward_list(x_or_x_list, rope_list=rope_or_rope_list)
        else:
            raise AssertionError


# In[11]:


norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
}

class MuMEncoder(nn.Module):
    def __init__(
        self,
        *,
        image_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        n_storage_tokens: int = 0,
        norm_layer: str = "layernorm",
        gradient_checkpointing: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        norm_layer_cls = norm_layer_dict[norm_layer]
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.gradient_checkpointing = gradient_checkpointing
        self.patch_embed = PatchEmbed(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            device=device,
        )
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, embed_dim, device=device) * 0.02
        )  # CLS
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(
                torch.empty(1, n_storage_tokens, embed_dim, device=device)
            )  # 额外可学习token
        block_cls = partial(
            SelfAttentionBlock,
            ffn_ratio=4.0,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            drop_path=0.0,
            norm_layer=norm_layer_cls,
            act_layer=nn.GELU,
            ffn_layer=Mlp,
            init_values=None,
            mask_k_bias=False,
            device=device,
        )
        self.blocks = nn.ModuleList(
            [block_cls(dim=embed_dim, num_heads=num_heads) for i in range(depth)]
        )
        self.norm = norm_layer_cls(embed_dim)

    def init_weights(self):
            self.rope_embed._init_weights()
            nn.init.normal_(self.cls_token, std=0.02)
            if self.n_storage_tokens > 0:
                nn.init.normal_(self.storage_tokens, std=0.02)
            named_apply(init_weights_vit, self)

    # 随机掩码
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    # [B,C,H,W,C]
    def forward(self, x, mask_ratio, return_all_blocks=False):
        # embed patches
        SB, C_in, H, W = x.shape
        x = self.patch_embed(x)  # [B, H,W, C]
        rope_sincos = self.rope_embed(H=x.shape[1], W=x.shape[2])  # 位置编码
        x = x.flatten(1, 2)  # [SB, L, C], with L=H*W

        # masking: length -> length * mask_ratio
        if not return_all_blocks:
            x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)

            # Let's just drop the masked patches in the rope
            sin, cos = rope_sincos
            sin_vis, cos_vis = sin[ids_keep], cos[ids_keep]  # [B, N_vis, D_head]
            sin_vis, cos_vis = sin_vis.unsqueeze(1).repeat(
                1, self.num_heads, 1, 1
            ), cos_vis.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            rope_sincos = (sin_vis, cos_vis)

        # append cls token and storage tokens
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens.expand(x.shape[0], -1, -1)
            x = torch.cat((storage_tokens, x), dim=1)

        # apply Transformer blocks
        if return_all_blocks:
            out = []
            for blk in self.blocks:
                if self.gradient_checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(
                        blk, x, rope_sincos, use_reentrant=False
                    )
                else:
                    x = blk(x, rope_sincos)
                out.append(x)
            return out
        else:
            for blk in self.blocks:
                if self.gradient_checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(
                        blk, x, rope_sincos, use_reentrant=False
                    )
                else:
                    x = blk(x, rope_sincos)
            x = self.norm(x)
        return x, mask, ids_restore


# In[12]:


class MuMDecoder(nn.Module):
    def __init__(
        self,
        *,
        image_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        encoder_embed_dim: int = 1024,
        decoder_embed_dim: int = 512,
        depth: int = 24,
        num_heads: int = 16,
        n_storage_tokens: int = 0,
        device: Optional[torch.device] = None,
        norm_layer: str = "layernorm",
        norm_pix_loss: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        norm_layer_cls = norm_layer_dict[norm_layer]
        self.image_size = image_size
        self.patch_size = patch_size
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.n_storage_tokens = n_storage_tokens
        self.norm_pix_loss = norm_pix_loss
        self.gradient_checkpointing = gradient_checkpointing
        self.decoder_embed = nn.Linear(
            encoder_embed_dim, decoder_embed_dim, bias=True, device=device
        )
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, decoder_embed_dim, device=device)
        )
        self.rope_embed = RopePositionEmbedding(
            embed_dim=decoder_embed_dim,
            num_heads=num_heads,
            device=device,
        )
        block_cls = partial(
            SelfAttentionBlock,
            ffn_ratio=4.0,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            drop_path=0.0,
            norm_layer=norm_layer_cls,
            act_layer=nn.GELU,
            ffn_layer=Mlp,
            init_values=None,
            mask_k_bias=False,
            device=device,
        )
        self.decoder_frame_blocks = nn.ModuleList(
            [  # 帧内注意力
                block_cls(dim=decoder_embed_dim, num_heads=num_heads)
                for i in range(depth // 2)
            ]
        )

        self.decoder_global_blocks = nn.ModuleList(
            [  # 帧间注意力
                block_cls(dim=decoder_embed_dim, num_heads=num_heads)
                for i in range(depth // 2)
            ]
        )

        self.decoder_norm = norm_layer_cls(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True, device=device
        )

    # 将图像分割为补丁，用于损失计算
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h, w = imgs.shape[2] // p, imgs.shape[3] // p 
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    # 将预测的补丁重建为图像，用于可视化
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * 3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    def init_weights(self):
            self.rope_embed._init_weights()
            nn.init.normal_(self.mask_token, std=.02)
            named_apply(init_weights_vit, self)

    
    def forward_decoder(self, x, ids_restore, B: int, S: int, H=None, W=None):
        # embed tokens
        num_patches_h, num_patches_w = H // self.patch_size, W // self.patch_size
        x = self.decoder_embed(x)

        sin, cos = self.rope_embed(H=num_patches_h, W=num_patches_w)
        rope_sincos_frame = (sin, cos)
        pos_special = (
            torch.zeros(1 + self.n_storage_tokens, sin.shape[-1])
            .to(sin.device)
            .to(sin.dtype)
        )
        sin, cos = torch.cat([pos_special, sin]), torch.cat([pos_special, cos])
        sin, cos = sin.repeat(S, 1), cos.repeat(S, 1)
        rope_sincos_global = (sin, cos)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        _, P, C = x.shape

        # apply alternating attention
        for frame_block, global_block in zip(
            self.decoder_frame_blocks, self.decoder_global_blocks
        ):
            # Frame-wise attention
            if x.shape != (B * S, P, C):
                x = x.view(B, S, P, C).view(B * S, P, C)

            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    frame_block, x, rope_sincos_frame, use_reentrant=False
                )
            else:
                x = frame_block(x, rope_sincos_frame)

            # Global attention
            x = x.view(B, S, P, C).view(B, S * P, C)

            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    global_block, x, rope_sincos_global, use_reentrant=False
                )
            else:
                x = global_block(x, rope_sincos_global)

        x = x.view(B, S, P, C).view(B * S, P, C)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, latent, mask, ids_restore):
        B, S, C_in, H, W = imgs.shape
        imgs = imgs.view(B * S, C_in, H, W)  # [B*S, C, H, W]
        pred = self.forward_decoder(
            latent, ids_restore, B, S, H=H, W=W
        )  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


# In[13]:


class MuM(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        norm_pix_loss: bool = True,
        norm_layer: str = "layernorm",
        n_storage_tokens: int = 0,
        device: Optional[Any] = None,
        gradient_checkpointing: bool = False,
        **ignored_kwargs,
    ):
        super().__init__()
        self.encoder = MuMEncoder(
            image_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            n_storage_tokens=n_storage_tokens,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.decoder = MuMDecoder(
            image_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            encoder_embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            device=device,
            norm_layer=norm_layer,
            gradient_checkpointing=gradient_checkpointing,
            n_storage_tokens=n_storage_tokens,
            norm_pix_loss=norm_pix_loss
        )

    def init_weights(self):
        self.encoder.init_weights()
        self.decoder.init_weights()

    def forward(self, imgs, mask_ratio=0.75, return_all_blocks=False):
        """
        imgs: [B, S, C, H, W] — 包含 S 帧的多帧输入
        """
        B, S, C, H, W = imgs.shape
        imgs_flat = imgs.reshape(B * S, C, H, W)  # [B*S, C, H, W] -> encoder

        latent, mask, ids_restore = self.encoder(
            imgs_flat, mask_ratio=mask_ratio, return_all_blocks=return_all_blocks
        )
        if return_all_blocks:
            return latent
        else:
            # decoder 需要 [B, S, C, H, W] 以恢复帧间结构
            loss, pred, mask = self.decoder(imgs, latent, mask, ids_restore)
            return loss, pred, mask

    #只用encoder提取特征
    def forward_encoder(self, imgs, return_all_blocks=True):
        """
        imgs: [B, S, C, H, W] 或 [B*S, C, H, W]
        """
        if imgs.dim() == 5:
            B, S, C, H, W = imgs.shape
            imgs = imgs.reshape(B * S, C, H, W)
        latent, mask, ids_restore = self.encoder(
            imgs,  return_all_blocks=return_all_blocks
        )
        return latent, mask, ids_restore


# In[14]:


def vit_base(patch_size=16, **kwargs):
    model = MuM(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        **kwargs,
    )
    return model


def vit_small(patch_size=16, **kwargs):
    model = MuM(
        patch_size=patch_size,
        embed_dim=384,
        depth=9,
        num_heads=6,
        decoder_embed_dim=256,
        decoder_depth=6,
        decoder_num_heads=8,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, **kwargs):
    model = MuM(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        **kwargs,
    )
    return model


# In[15]:


class BlendedMVSDataset(Dataset):
    """
    BlendedMVS 数据集: 每个场景随机采样 S 帧连续图像（S 在 [S_min, S_max] 中随机）.

    每帧图像形状: [3, H, W], 返回形状: [S, 3, H, W]
    """
    def __init__(
        self,
        root: str,
        S_min: int = 3,
        S_max: int = 5,
        img_size: int = 224,
        train: bool = True,
    ):
        self.root = root
        self.S_min = S_min
        self.S_max = S_max
        self.img_size = img_size
        self.train = train

        # 扫描所有场景
        scene_dirs = sorted(glob.glob(os.path.join(root, "*", "blended_images")))
        assert len(scene_dirs) > 0, f"在 {root} 中未找到场景文件夹"

        # 每个场景保存其所有 unmasked 图片路径
        self.scene_paths = []  # list[list[str]]
        for scene_dir in scene_dirs:
            paths = sorted(glob.glob(os.path.join(scene_dir, "*.jpg")))
            paths = [p for p in paths if "_masked" not in p]
            if len(paths) >= S_min:  # 至少要有 S_min 张图
                self.scene_paths.append(paths)

        # 每个场景复制 4 次（随机 S 帧 + 随机裁剪确保多样性）
        self.num_scenes = len(self.scene_paths)
        self.num_samples = self.num_scenes * 4

        print(f"[BlendedMVSDataset] {self.num_scenes} 个场景 × 4, "
              f"S∈[{S_min},{S_max}], {self.num_samples} 个样本")

        # 数据增强
        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    img_size, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(int(img_size * 1.1), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        paths = self.scene_paths[idx % self.num_scenes]

        # 随机选 S 张图
        S = random.randint(self.S_min, min(self.S_max, len(paths)))
        chosen = random.sample(paths, S)

        imgs = []
        for path in chosen:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
            imgs.append(img)

        return torch.stack(imgs, dim=0)  # [S, 3, H, W]


def pad_collate_fn(batch):
    """
    将 batch 中不同 S 的样本 pad 到统一的 S_max（本 batch 内最大值）。
    不足的用最后一帧重复填充。
    """
    max_S = max(x.shape[0] for x in batch)
    padded = []
    for x in batch:
        S = x.shape[0]
        if S < max_S:
            # 重复最后一帧补足
            pad = x[-1:].repeat(max_S - S, 1, 1, 1)
            x = torch.cat([x, pad], dim=0)
        padded.append(x)
    return torch.stack(padded, dim=0)  # [B, max_S, 3, H, W]


# In[16]:


# 快速测试（仅在非 DDP 或主进程运行）
_local_rank = os.environ.get("LOCAL_RANK", None)
if _local_rank is None or int(_local_rank) == 0:
    root = "/data/data_taohy/datasets/BlendedMVS"
    ds = BlendedMVSDataset(root, S_min=3, S_max=5, img_size=224, train=True)
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0, collate_fn=pad_collate_fn)
    batch = next(iter(loader))  # [4, S, 3, 224, 224], S ∈ [2,4] 统一到 max_S
    print(f"batch shape: {batch.shape} (B={batch.shape[0]}, S={batch.shape[1]})")

    model = vit_small(patch_size=16, img_size=224, norm_pix_loss=False, norm_layer="layernormbf16")
    model.init_weights()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"参数数量: {n_params:.1f}M")

    with torch.no_grad():
        loss, pred, mask = model(batch, mask_ratio=0.75)
        print(f"✅ Forward pass 成功! loss={loss.item():.6f}, pred={pred.shape}, mask_rate={mask.float().mean().item():.3f}")

    # 释放测试用的模型和 batch，给训练腾显存
    del model, batch, loader, ds


# In[17]:


def train(
    data_root: str = "/data/data_taohy/datasets/BlendedMVS",
    output_dir: str = "/data/data_taohy/modelReShow/mum/checkpoints",
    *,
    # 模型
    model_name: str = "vit_small",
    patch_size: int = 16,
    img_size: int = 224,
    norm_pix_loss: bool = True,
    norm_layer: str = "layernormbf16",
    # 数据
    S_min: int = 3,
    S_max: int = 5,
    # 训练超参
    total_steps: int = 15000,
    warmup_steps: int = 1000,
    batch_size: int = 2,
    accum_steps: int = 4,
    lr_base: float = 1e-2,
    weight_decay: float = 0.05,
    betas: tuple = (0.9, 0.95),
    mask_ratio: float = 0.75,
    clip_grad_norm: float = 1.0,
    # 日志 / 保存
    save_every_steps: int = 1800,
    log_every: int = 50,
    # wandb
    use_wandb: bool = True,
    wandb_project: str = "MuM",
    wandb_run_name: str = None,
    # 设备 / 数据加载
    device: str = None,
    num_workers: int = 8,
    # 恢复训练
    resume_ckpt: str = None,
    # 混合精度
    use_amp: bool = True,
    amp_dtype: str = "bf16",  # "bf16" 或 "fp16"
    # DDP 多卡参数
    local_rank: int = 0,
    world_size: int = 1,
):
    """
    MuM (Masked unsupervised Model) 训练函数 — ViT-Large + 500K steps.

    参数
    ----
    model_name : "vit_base" 或 "vit_large"
    norm_layer : "layernorm" (eps=1e-6) 或 "layernormbf16" (eps=1e-5)
    S_min / S_max : 每样本随机采样帧数范围
    total_steps : 总优化步数
    warmup_steps : 线性 warmup 步数
    save_every_steps : 每隔多少步保存一次 checkpoint
    resume_ckpt : 从指定 checkpoint 恢复训练（路径指向 .pth 文件）
    local_rank : DDP 当前进程 rank（单卡时保持 0）
    world_size : DDP 总进程数（单卡时保持 1）
    
    多卡启动方式:
        terminal:  torchrun --nproc_per_node=4 script.py
        notebook:  mp.spawn(train_ddp_entry, args=(...), nprocs=4)
    """
    # ---------- DDP 初始化 ----------
    # 自动检测 torchrun 环境（通过 LOCAL_RANK 环境变量）
    local_rank_env = os.environ.get("LOCAL_RANK", None)
    if local_rank_env is not None:
        # torchrun 启动，自动初始化 DDP
        ddp = True
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(local_rank_env)
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        ddp = world_size > 1
        if ddp and not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
            local_rank = dist.get_rank()
            world_size = dist.get_world_size()
    
    is_main = (not ddp) or (dist.get_rank() == 0)

    if ddp:
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    elif device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if is_main:
        os.makedirs(output_dir, exist_ok=True)

    # ---------- 生成本次运行的唯一标识（时间戳），防止覆盖 ----------
    run_tag = datetime.now().strftime("%m%d-%H%M%S")

    # ---------- wandb ----------
    if use_wandb and is_main:
        effective_bs = batch_size * accum_steps * world_size
        run_name = wandb_run_name or f"mum_{model_name}_S{S_min}-{S_max}_steps{total_steps//1000}k"
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "model": f"MuM-{model_name}",
                "patch_size": patch_size,
                "img_size": img_size,
                "norm_layer": norm_layer,
                "S_min": S_min,
                "S_max": S_max,
                "total_steps": total_steps,
                "warmup_steps": warmup_steps,
                "batch_size": batch_size,
                "accum_steps": accum_steps,
                "effective_batch": effective_bs,
                "lr_base": lr_base,
                "weight_decay": weight_decay,
                "mask_ratio": mask_ratio,
                "clip_grad_norm": clip_grad_norm,
                "norm_pix_loss": norm_pix_loss,
                "device": device,
                "world_size": world_size,
            },
        )

    # ---------- 数据 ----------
    train_ds = BlendedMVSDataset(data_root, S_min=S_min, S_max=S_max, img_size=img_size, train=True)
    sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=local_rank if ddp else 0, shuffle=True
    ) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=pad_collate_fn,
    )

    # ---------- 模型 ----------
    model_factories = {"vit_small": vit_small, "vit_base": vit_base, "vit_large": vit_large}
    model_factory = model_factories[model_name]
    model = model_factory(
        patch_size=patch_size, img_size=img_size,
        norm_pix_loss=norm_pix_loss, norm_layer=norm_layer,
    )
    model.init_weights()
    model.to(device)

    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if use_wandb and is_main:
        raw_model = model.module if ddp else model
        wandb.watch(raw_model, log="gradients", log_freq=100)

    # ---------- 优化器 ----------
    param_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or "norm" in name:
            param_groups.append({"params": param, "weight_decay": 0.0})
        else:
            param_groups.append({"params": param, "weight_decay": weight_decay})

    optimizer = AdamW(param_groups, lr=lr_base, betas=betas)

    # Step-based LR schedule: linear warmup → cosine decay
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ---------- 恢复训练 ----------
    step_count = 0
    if resume_ckpt is not None:
        ckpt = torch.load(resume_ckpt, map_location=device)
        raw_model = model.module if ddp else model
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step_count = ckpt.get("step", 0)
        # 将 scheduler 快进到当前步数（屏蔽 optimizer-before-scheduler 警告，此处无害）
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Detected call of.*scheduler.step")
            for _ in range(step_count):
                scheduler.step()
        if is_main:
            print(f"🔄 从 step {step_count} 恢复训练 (loss={ckpt['loss']:.4f})")

    # ---------- 混合精度 ----------
    if use_amp:
        amp_enabled = amp_dtype == "bf16" or amp_dtype == "fp16"
        amp_dtype_t = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == "fp16"))
        autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype_t, enabled=amp_enabled)
    else:
        amp_enabled = False
        autocast_ctx = nullcontext()

    # ---------- 训练 ----------
    if is_main:
        amp_info = f", AMP={amp_dtype}" if use_amp else ", AMP=off"
        print(f"设备: {device}, GPU数: {world_size}, 场景数: {train_ds.num_scenes}, "
              f"样本数: {len(train_ds)}, S∈[{S_min},{S_max}]{amp_info}")
        print(f"有效 batch: {batch_size * accum_steps * world_size}, "
              f"总优化步数: {total_steps}, warmup: {warmup_steps}")
        print(f"模型: {model_name}, norm_layer: {norm_layer}")

    window_loss = deque(maxlen=log_every)  # 滑动窗口，用于打印平滑 loss
    step_start = time.time()
    epoch = 0
    while step_count < total_steps:
        epoch += 1
        if ddp:
            sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device, non_blocking=True)

            # DDP: 只在最后一个累积步同步梯度，中间步骤不同步以提升效率
            is_accum_step = (batch_idx + 1) % accum_steps == 0
            sync_ctx = nullcontext() if is_accum_step or not ddp else model.no_sync()
            with sync_ctx:
                with autocast_ctx:
                    loss, pred, mask = model(batch, mask_ratio=mask_ratio)
                    loss = loss / accum_steps
                if amp_dtype == "fp16":
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if is_accum_step:
                if amp_dtype == "fp16":
                    scaler.unscale_(optimizer)
                if clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                if amp_dtype == "fp16":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step_count += 1

                # 记录当前 step 的 loss（反向缩放回原始尺度）
                window_loss.append(loss.item() * accum_steps)

                if step_count % log_every == 0 and is_main:
                    lr_now = optimizer.param_groups[0]["lr"]
                    avg_loss = sum(window_loss) / len(window_loss)
                    elapsed = time.time() - step_start
                    print(f"[step {step_count:6d}/{total_steps}] "
                          f"loss={avg_loss:.4f} | lr={lr_now:.2e} | {elapsed:.1f}s")
                    step_start = time.time()
                    if use_wandb:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/lr": lr_now,
                            "train/step": step_count,
                        })

                # 保存 checkpoint（仅主进程）
                if step_count % save_every_steps == 0 and is_main:
                    ckpt_path = os.path.join(output_dir, f"mum_{run_tag}_step{step_count}.pth")
                    raw_model = model.module if ddp else model
                    torch.save({
                        "step": step_count,
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": sum(window_loss) / max(1, len(window_loss)),
                    }, ckpt_path)
                    print(f"  💾 Checkpoint → {ckpt_path}")
                    if use_wandb:
                        wandb.save(ckpt_path)

                if step_count >= total_steps:
                    break

    # 最终保存（仅主进程）
    if is_main:
        raw_model = model.module if ddp else model
        final_path = os.path.join(output_dir, f"mum_{run_tag}_final_step{step_count}.pth")
        torch.save({
            "step": step_count,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": sum(window_loss) / max(1, len(window_loss)),
        }, final_path)
        print(f"  💾 Final → {final_path}")
        if use_wandb:
            wandb.save(final_path)
            wandb.finish()
        print("🎉 训练完成!")

    if ddp:
        dist.destroy_process_group()

    return model.module if ddp else model


# In[18]:


# ============================================================
# 4 卡 DDP 训练 — ViT-Large, 500K steps, S∈[2,12]
# ============================================================
# 方式 1（推荐）：将 notebook 导出为 .py 文件后在终端用 torchrun 启动：
#   jupyter nbconvert --to script mum.ipynb --output train.py
#   torchrun --nproc_per_node=4 train.py
#
# 方式 2：直接在 notebook 中使用 mp.spawn（需要确保所有依赖在子进程中可导入）
# 如果方式 1 不方便，取消下方注释即可：
#
# import torch.multiprocessing as mp
# 
# def ddp_entry(local_rank, world_size):
#     """DDP 训练入口，由 mp.spawn 在每个 GPU 上调用"""
#     train(
#         data_root="/data/data_taohy/datasets/BlendedMVS",
#         output_dir="/data/data_taohy/modelReShow/mum/checkpoints",
#         local_rank=local_rank,
#         world_size=world_size,
#         # 如需从 checkpoint 恢复，取消此行注释：
#         # resume_ckpt="/data/data_taohy/modelReShow/mum/checkpoints/mum_0701-143025_step50000.pth",
#     )
# 
# n_gpus = 4
# mp.spawn(ddp_entry, args=(n_gpus,), nprocs=n_gpus)

# 单卡模式（保留兼容）：
# model = train(
#     data_root="/data/data_taohy/datasets/BlendedMVS",
#     output_dir="/data/data_taohy/modelReShow/mum/checkpoints",
# )

if __name__ == "__main__":
    # 使用 ViT-Small (~22M 参数) 从头训练，适合 452 样本的小数据集
    model = train(
        data_root="/data/data_taohy/datasets/BlendedMVS",
        output_dir="/data/data_taohy/modelReShow/mum/checkpoints",
        model_name="vit_small",
        total_steps=15000,
        warmup_steps=1000,
        weight_decay=0.1,
        mask_ratio=0.6,
    )