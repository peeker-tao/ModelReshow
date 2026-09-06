# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vit import PatchEmbed, Block
from models.pos_embed import get_2d_sincos_pos_embed


class TinyMIMViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, drop_path=0.1,
                 embed_dim=1024, depth=24, num_heads=16,last_heads=12,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        # --------------------------------------------------------------------------
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.last_heads = last_heads
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop_path=drop_path, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth-1)]+[Block(embed_dim, self.last_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_embed(self, h, w):
        """Interpolate the fixed sin-cos position embedding to a (h x w) patch grid."""
        num_patches = h * w
        if num_patches == self.pos_embed.shape[1] - 1:
            return self.pos_embed
        dim = self.pos_embed.shape[-1]
        cls_pos_embed = self.pos_embed[:, :1, :]
        patch_pos_embed = self.pos_embed[:, 1:, :]  # (1, N0, dim)
        h0 = w0 = int(patch_pos_embed.shape[1] ** 0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, h0, w0, dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(patch_pos_embed, size=(h, w), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, num_patches, dim)
        return torch.cat((cls_pos_embed, patch_pos_embed), dim=1)

    def forward_encoder(self, x):
        # embed patches
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        h = H // self.patch_embed.patch_size[0]
        w = W // self.patch_embed.patch_size[1]

        # add pos embed w/o cls token (dynamically interpolated for arbitrary sizes)
        pos_embed = self.interpolate_pos_embed(h, w).to(x.device)
        x = x + pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        x_out = []
        x_out.append(x)
        for blk in self.blocks:
            x = blk(x)
            x_out.append(x)
        return x_out

    def forward_kd_loss(self, pred, teacher_out):
        loss = nn.KLDivLoss(reduction="none")(pred.log(), teacher_out).sum(-1)
        return loss.mean()

    def forward(self, imgs):

        return self.forward_encoder(imgs)


def tinymim_vit_tiny_patch16(**kwargs):
    model = TinyMIMViT(
        patch_size=16, embed_dim=192, depth=12, num_heads=6, drop_path=0.1,last_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def tinymim_vit_small_patch16(**kwargs):
    model = TinyMIMViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,drop_path=0.1,last_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def tinymim_vit_base_patch16(**kwargs):
    model = TinyMIMViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, drop_path=0.1,last_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


