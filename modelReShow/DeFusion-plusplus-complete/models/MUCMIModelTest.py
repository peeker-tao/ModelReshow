"""DeFusion++ inference model (multi-modal fusion test).

This is a clean reconstruction of the DeFusion++ fusion network, derived from
the historical ``MMUCMIModel.py`` (commit 136f957) of the official repository.

It only keeps the inference-relevant components:
    * a TinyMIM-ViT (tiny) encoder shared by both source images,
    * the common / unique decomposition (CUD) decoders,
    * the fusion decoder.

Training-only components (the two MAE teachers, the mask token, and the masked
feature modeling decoder) are removed.  ``forward`` takes a pair of images
(IR / VI, both RGB in [0, 1]) and returns four images in [0, 1]:

    (common_part, upper_part, lower_part, fusion_part)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from models.tinymim import tinymim_vit_tiny_patch16
from models.vit import Block
from models.multiAtten import TransformerAttenBlockVpaper as AttenBlock


class MUCMIMNetTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = tinymim_vit_tiny_patch16()

        decoder_embed_dim = 192
        decoder_img_dim = 768  # patch_size ** 2 * in_chans = 16*16*3
        decoder_num_heads = 16
        mlp_ratio = 4.
        norm_layer = nn.LayerNorm

        self.enc_norm1 = nn.LayerNorm(decoder_embed_dim)
        self.enc_norm2 = nn.LayerNorm(decoder_embed_dim)

        self.recon_blocks_mim_encoder = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio,
                  qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for _ in range(2)
        ])

        # common part decoder
        self.decoder_common_blocks = nn.ModuleList([
            AttenBlock(decoder_embed_dim, decoder_num_heads, common=True,
                       dim_feedforward=int(mlp_ratio * decoder_embed_dim)),
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio,
                  qkv_bias=True, qk_scale=None, norm_layer=norm_layer),
        ])
        self.decoder_common_blocks.append(nn.Linear(decoder_embed_dim, decoder_img_dim))
        self.decode_common_skipconn = AttenBlock(
            decoder_embed_dim, decoder_num_heads, common=True,
            dim_feedforward=int(mlp_ratio * decoder_embed_dim))

        # unique part decoder
        self.decoder_unique_blocks = nn.ModuleList([
            AttenBlock(decoder_embed_dim, decoder_num_heads, common=False,
                       dim_feedforward=int(mlp_ratio * decoder_embed_dim)),
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio,
                  qkv_bias=True, qk_scale=None, norm_layer=norm_layer),
        ])
        self.decoder_unique_blocks.append(nn.Linear(decoder_embed_dim, decoder_img_dim))
        self.decoder_unique_residual = Block(
            decoder_embed_dim, decoder_num_heads, mlp_ratio,
            qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        self.decoder_unique_skipconn = AttenBlock(
            decoder_embed_dim, decoder_num_heads, common=False,
            dim_feedforward=int(mlp_ratio * decoder_embed_dim))
        self.decoder_unique_residual_skipconn = Block(
            decoder_embed_dim, decoder_num_heads, mlp_ratio,
            qkv_bias=True, qk_scale=None, norm_layer=norm_layer)

        # fusion decoder
        self.decoder_fuse_blocks = nn.ModuleList([
            AttenBlock(decoder_embed_dim, decoder_num_heads, common=False,
                       dim_feedforward=int(mlp_ratio * decoder_embed_dim)),
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio,
                  qkv_bias=True, qk_scale=None, norm_layer=norm_layer),
        ])
        self.decoder_fuse_blocks.append(nn.Linear(decoder_embed_dim, decoder_img_dim))
        self.decoder_fuse_skipconn = AttenBlock(
            decoder_embed_dim, decoder_num_heads, common=False,
            dim_feedforward=int(mlp_ratio * decoder_embed_dim))

        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]

    def unpatchify(self, x, h, w, c=3, p=16):
        """x: (N, L, p*p*c) -> imgs: (N, c, h*p, w*p)"""
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def _denormalize(self, img):
        mean = torch.tensor(self.normalize_mean, device=img.device).view(1, 3, 1, 1)
        std = torch.tensor(self.normalize_std, device=img.device).view(1, 3, 1, 1)
        return img * std + mean

    def forward(self, img1, img2):
        B, C, H, W = img1.shape
        h, w = H // 16, W // 16

        img1 = normalize(img1, self.normalize_mean, self.normalize_std)
        img2 = normalize(img2, self.normalize_mean, self.normalize_std)

        enc_feas1 = self.encoder(img1)
        enc_feas2 = self.encoder(img2)

        residual_fea1 = self.enc_norm1(enc_feas1[0]) + self.enc_norm2(enc_feas1[1])
        residual_fea2 = self.enc_norm1(enc_feas2[0]) + self.enc_norm2(enc_feas2[1])

        enc_fea1 = enc_feas1[-1]
        enc_fea2 = enc_feas2[-1]

        # ---- common decomposition ----
        com_img = self.decoder_common_blocks[0](enc_fea1, enc_fea2)
        residual_com_img = self.decode_common_skipconn(residual_fea1, residual_fea2)

        # ---- unique decomposition ----
        uni_img2 = self.decoder_unique_blocks[0](enc_fea1, enc_fea2) + self.decoder_unique_residual(enc_fea2)
        residual_uni_img2 = self.decoder_unique_skipconn(residual_fea1, residual_fea2) + \
            self.decoder_unique_residual_skipconn(residual_fea2)

        uni_img1 = self.decoder_unique_blocks[0](enc_fea2, enc_fea1) + self.decoder_unique_residual(enc_fea1)
        residual_uni_img1 = self.decoder_unique_skipconn(residual_fea2, residual_fea1) + \
            self.decoder_unique_residual_skipconn(residual_fea1)

        # ---- fusion ----
        fuse_img = self.decoder_fuse_blocks[0](com_img, uni_img1) + self.decoder_fuse_blocks[0](com_img, uni_img2)
        residual_fuse_img = self.decoder_fuse_skipconn(residual_com_img, residual_uni_img1) + \
            self.decoder_fuse_skipconn(residual_com_img, residual_uni_img2)

        com_img = com_img + residual_com_img
        uni_img2 = uni_img2 + residual_uni_img2
        uni_img1 = uni_img1 + residual_uni_img1

        fuse_img = fuse_img + residual_fuse_img
        fuse_main = fuse_img

        for blk in self.recon_blocks_mim_encoder:
            fuse_img = blk(fuse_img)

        for blk in self.decoder_common_blocks[1:]:
            com_img = blk(com_img)
        for blk in self.decoder_unique_blocks[1:]:
            uni_img1 = blk(uni_img1)
            uni_img2 = blk(uni_img2)

        fuse_img = fuse_img + fuse_main
        for blk in self.decoder_fuse_blocks[1:]:
            fuse_img = blk(fuse_img)

        # decode token sequence back to image
        com_img = self.unpatchify(com_img[:, 1:, :], h, w)
        uni_img1 = self.unpatchify(uni_img1[:, 1:, :], h, w)
        uni_img2 = self.unpatchify(uni_img2[:, 1:, :], h, w)
        fuse_img = self.unpatchify(fuse_img[:, 1:, :], h, w)

        # back to [0, 1]
        com_img = self._denormalize(com_img)
        uni_img1 = self._denormalize(uni_img1)
        uni_img2 = self._denormalize(uni_img2)
        fuse_img = self._denormalize(fuse_img)

        return com_img, uni_img1, uni_img2, fuse_img
