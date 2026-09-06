"""DeFusion++ multi-modality self-supervised (MCUD) training model.

Clean reconstruction of the historical ``MMUCMIModel.py`` (commit 136f957).

Stage 2 (MCUD, "Multi-modal Common-Unique Decomposition") aligns the latent
representations of two modalities (IR / VI) with two frozen MAE teachers.

Bugs fixed relative to the original:
    * the undefined ``fuse_main`` residual variable,
    * image outputs are de-normalized back to [0, 1] so they match the
      [0, 1] targets used by the losses.

``forward(img1, img2, modality)``:
    * ``modality == 'irvis'`` -> returns latent predictions for MCUD:
        (modality_gt1, modality_gt2, modality_predict1, modality_predict2,
         com_img_w1, com_img_w2)
    * otherwise -> returns images in [0, 1] for CUD:
        (rec_img2, rec_img2, com_img, uni_img1, uni_img2, fuse_img)
"""
import os
import torch
import torch.nn as nn
from torchvision.transforms.functional import normalize

from models.tinymim import tinymim_vit_tiny_patch16
from models.models_mae import mae_vit_base_patch16
from models.vit import Block
from models.multiAtten import TransformerAttenBlockVpaper as AttenBlock
from utils.pos_embed import get_2d_sincos_pos_embed


class MUCMIMNet(nn.Module):
    def __init__(self, teacher1_pth=None, teacher2_pth=None, use_teacher=True):
        super(MUCMIMNet, self).__init__()
        self.use_teacher = use_teacher
        self.encoder = tinymim_vit_tiny_patch16()

        # ---- frozen MAE teachers (modality latents) ----
        # 仅 MCUD 阶段需要；纯 CUD 阶段 use_teacher=False 不创建，省显存
        # （约 2×86M 参数）。阶段2 init_from 阶段1 权重时 strict=False，
        # 缺失的只有 teacher keys，保留此处加载的预训练 MAE 权重。
        if use_teacher:
            self.model_encoder1 = mae_vit_base_patch16()
            self.model_encoder2 = mae_vit_base_patch16()
            if teacher1_pth and os.path.exists(teacher1_pth):
                self._load_teacher(self.model_encoder1, teacher1_pth)
            else:
                print('Warning: teacher1 weights not found, using random init: {}'.format(teacher1_pth))
            if teacher2_pth and os.path.exists(teacher2_pth):
                self._load_teacher(self.model_encoder2, teacher2_pth)
            else:
                print('Warning: teacher2 weights not found, using random init: {}'.format(teacher2_pth))
            for _, p in self.model_encoder1.named_parameters():
                p.requires_grad = False
            for _, p in self.model_encoder2.named_parameters():
                p.requires_grad = False

        decoder_embed_dim = 192
        decoder_img_dim = 768
        decoder_num_heads = 16
        pretrain_model_dim = 768
        mlp_ratio = 4.
        norm_layer = nn.LayerNorm

        self.enc_norm1 = nn.LayerNorm(decoder_embed_dim)
        self.enc_norm2 = nn.LayerNorm(decoder_embed_dim)
        self.recon_blocks_mim_encoder = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio,
                  qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for _ in range(2)
        ])

        self.decoder_common_blocks = nn.ModuleList([
            AttenBlock(decoder_embed_dim, decoder_num_heads, common=True,
                       dim_feedforward=int(mlp_ratio * decoder_embed_dim)),
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio,
                  qkv_bias=True, qk_scale=None, norm_layer=norm_layer)])
        self.decoder_common_blocks.append(nn.Linear(decoder_embed_dim, decoder_img_dim))
        self.decode_common_skipconn = AttenBlock(
            decoder_embed_dim, decoder_num_heads, common=True,
            dim_feedforward=int(mlp_ratio * decoder_embed_dim))

        self.decoder_unique_blocks = nn.ModuleList([
            AttenBlock(decoder_embed_dim, decoder_num_heads, common=False,
                       dim_feedforward=int(mlp_ratio * decoder_embed_dim)),
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio,
                  qkv_bias=True, qk_scale=None, norm_layer=norm_layer)])
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

        self.latent_predict1 = nn.Linear(decoder_embed_dim, pretrain_model_dim, bias=True)
        self.latent_predict2 = nn.Linear(decoder_embed_dim, pretrain_model_dim, bias=True)
        self.mm_common_norm = nn.LayerNorm(decoder_embed_dim, elementwise_affine=False)
        self.mm_unique_norm1 = nn.LayerNorm(decoder_embed_dim)
        self.mm_unique_norm2 = nn.LayerNorm(decoder_embed_dim)

        self.decoder_fuse_blocks = nn.ModuleList([
            AttenBlock(decoder_embed_dim, decoder_num_heads, common=False,
                       dim_feedforward=int(mlp_ratio * decoder_embed_dim)),
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio,
                  qkv_bias=True, qk_scale=None, norm_layer=norm_layer)])
        self.decoder_fuse_blocks.append(nn.Linear(decoder_embed_dim, decoder_img_dim))
        self.decoder_fuse_skipconn = AttenBlock(
            decoder_embed_dim, decoder_num_heads, common=False,
            dim_feedforward=int(mlp_ratio * decoder_embed_dim))

        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 196 + 1, decoder_embed_dim),
                                              requires_grad=False)
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.encoder.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        self.mim_decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio,
                  qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for _ in range(2)])
        self.mim_decoder_norm = norm_layer(decoder_embed_dim)
        self.mim_decoder_pred = nn.Linear(decoder_embed_dim, decoder_img_dim, bias=True)

    def _load_teacher(self, model, pth):
        checkpoint = torch.load(pth, map_location='cpu')
        print("Load teacher checkpoint from: %s" % pth)
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        model.load_state_dict(checkpoint, strict=False)

    def unpatchifyc(self, x, c=3, p=16):
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape((x.shape[0], c, h * p, h * p))

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_masked, ids_restore, ids_keep

    def align_masking(self, x, ids_keep):
        N, L, D = x.shape
        return torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    def forward_decoder(self, x, ids_restore):
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.mim_decoder_blocks:
            x = blk(x)
        x = self.mim_decoder_norm(x)
        x = self.mim_decoder_pred(x)
        return x

    def _denormalize(self, img):
        mean = torch.tensor(self.normalize_mean, device=img.device).view(1, 3, 1, 1)
        std = torch.tensor(self.normalize_std, device=img.device).view(1, 3, 1, 1)
        return img * std + mean

    def forward(self, img1, img2, modality=None):
        if modality == 'irvis':
            assert self.use_teacher, \
                'MCUD branch (modality="irvis") requires use_teacher=True (frozen MAE teachers).'
            img1 = normalize(img1, self.normalize_mean, self.normalize_std)
            img2 = normalize(img2, self.normalize_mean, self.normalize_std)

            enc_feas1 = self.encoder(img1)
            enc_feas2 = self.encoder(img2)

            with torch.no_grad():
                modality_gt1 = self.model_encoder1(img1)
                modality_gt2 = self.model_encoder2(img2)

            residual_fea1 = self.enc_norm1(enc_feas1[0]) + self.enc_norm2(enc_feas1[1])
            residual_fea2 = self.enc_norm1(enc_feas2[0]) + self.enc_norm2(enc_feas2[1])

            enc_fea1 = enc_feas1[-1]
            enc_fea2 = enc_feas2[-1]

            com_img_w1 = self.decoder_common_blocks[0](enc_fea1, enc_fea2)
            residual_com_img_w1 = self.decode_common_skipconn(residual_fea1, residual_fea2)

            com_img_w2 = self.decoder_common_blocks[0](enc_fea2, enc_fea1)
            residual_com_img_w2 = self.decode_common_skipconn(residual_fea2, residual_fea1)

            uni_img2 = self.decoder_unique_blocks[0](enc_fea1, enc_fea2) + self.decoder_unique_residual(enc_fea2)
            residual_uni_img2 = self.decoder_unique_skipconn(residual_fea1, residual_fea2) + \
                self.decoder_unique_residual_skipconn(residual_fea2)

            uni_img1 = self.decoder_unique_blocks[0](enc_fea2, enc_fea1) + self.decoder_unique_residual(enc_fea1)
            residual_uni_img1 = self.decoder_unique_skipconn(residual_fea2, residual_fea1) + \
                self.decoder_unique_residual_skipconn(residual_fea1)

            com_img_w1 = com_img_w1 + residual_com_img_w1
            com_img_w2 = com_img_w2 + residual_com_img_w2
            uni_img2 = uni_img2 + residual_uni_img2
            uni_img1 = uni_img1 + residual_uni_img1

            com_img_w1 = self.mm_common_norm(com_img_w1)
            com_img_w2 = self.mm_common_norm(com_img_w2)

            modality_predict1 = self.latent_predict1(self.mm_unique_norm1(uni_img1) + com_img_w1)
            modality_predict2 = self.latent_predict2(self.mm_unique_norm2(uni_img2) + com_img_w2)

            return modality_gt1, modality_gt2, modality_predict1, modality_predict2, com_img_w1, com_img_w2

        # ---- single-modality CUD branch ----
        img1 = normalize(img1, self.normalize_mean, self.normalize_std)
        img2 = normalize(img2, self.normalize_mean, self.normalize_std)

        enc_feas1 = self.encoder(img1)
        enc_feas2 = self.encoder(img2)
        residual_fea1 = self.enc_norm1(enc_feas1[0]) + self.enc_norm2(enc_feas1[1])
        residual_fea2 = self.enc_norm1(enc_feas2[0]) + self.enc_norm2(enc_feas2[1])

        enc_fea1 = enc_feas1[-1]
        enc_fea2 = enc_feas2[-1]

        if torch.rand(1).item() > 0.5:
            com_img = self.decoder_common_blocks[0](enc_fea1, enc_fea2)
            residual_com_img = self.decode_common_skipconn(residual_fea1, residual_fea2)
        else:
            com_img = self.decoder_common_blocks[0](enc_fea2, enc_fea1)
            residual_com_img = self.decode_common_skipconn(residual_fea2, residual_fea1)

        uni_img2 = self.decoder_unique_blocks[0](enc_fea1, enc_fea2) + self.decoder_unique_residual(enc_fea2)
        residual_uni_img2 = self.decoder_unique_skipconn(residual_fea1, residual_fea2) + \
            self.decoder_unique_residual_skipconn(residual_fea2)

        uni_img1 = self.decoder_unique_blocks[0](enc_fea2, enc_fea1) + self.decoder_unique_residual(enc_fea1)
        residual_uni_img1 = self.decoder_unique_skipconn(residual_fea2, residual_fea1) + \
            self.decoder_unique_residual_skipconn(residual_fea1)

        com_mimg, ids_restore, ids_keep = self.random_masking(com_img[:, 1:, :], mask_ratio=0.75)
        com_mimg = torch.cat([com_img[:, :1, :], com_mimg], dim=1)

        def align_and_concatenate(img, ids_keep):
            aligned_mimg = self.align_masking(img[:, 1:, :], ids_keep)
            return torch.cat([img[:, :1, :], aligned_mimg], dim=1)

        uni_mimg1 = align_and_concatenate(uni_img1, ids_keep)
        uni_mimg2 = align_and_concatenate(uni_img2, ids_keep)
        residual_com_mimg = align_and_concatenate(residual_com_img, ids_keep)
        residual_uni_mimg1 = align_and_concatenate(residual_uni_img1, ids_keep)
        residual_uni_mimg2 = align_and_concatenate(residual_uni_img2, ids_keep)

        fuse_img = self.decoder_fuse_blocks[0](com_img, uni_img1) + self.decoder_fuse_blocks[0](com_img, uni_img2)
        rec_img = self.decoder_fuse_blocks[0](com_mimg, uni_mimg1) + self.decoder_fuse_blocks[0](com_mimg, uni_mimg2)
        residual_fuse_img = self.decoder_fuse_skipconn(residual_com_img, residual_uni_img1) + \
            self.decoder_fuse_skipconn(residual_com_img, residual_uni_img2)
        residual_rec_img = self.decoder_fuse_skipconn(residual_com_mimg, residual_uni_mimg1) + \
            self.decoder_fuse_skipconn(residual_com_mimg, residual_uni_mimg2)

        com_img = com_img + residual_com_img
        uni_img2 = uni_img2 + residual_uni_img2
        uni_img1 = uni_img1 + residual_uni_img1

        fuse_img = fuse_img + residual_fuse_img
        fuse_main = fuse_img
        rec_img = rec_img + residual_rec_img

        for blk in self.recon_blocks_mim_encoder:
            rec_img = blk(rec_img)
            fuse_img = blk(fuse_img)

        for blk in self.decoder_common_blocks[1:]:
            com_img = blk(com_img)
        for blk in self.decoder_unique_blocks[1:]:
            uni_img1 = blk(uni_img1)
            uni_img2 = blk(uni_img2)

        fuse_img = fuse_img + fuse_main
        for blk in self.decoder_fuse_blocks[1:]:
            fuse_img = blk(fuse_img)

        rec_img = self.forward_decoder(rec_img, ids_restore)

        rec_img2 = self.unpatchifyc(rec_img[:, 1:, :], c=3, p=16)
        com_img = self.unpatchifyc(com_img[:, 1:, :], c=3, p=16)
        uni_img1 = self.unpatchifyc(uni_img1[:, 1:, :], c=3, p=16)
        uni_img2 = self.unpatchifyc(uni_img2[:, 1:, :], c=3, p=16)
        fuse_img = self.unpatchifyc(fuse_img[:, 1:, :], c=3, p=16)

        rec_img2 = self._denormalize(rec_img2)
        com_img = self._denormalize(com_img)
        uni_img1 = self._denormalize(uni_img1)
        uni_img2 = self._denormalize(uni_img2)
        fuse_img = self._denormalize(fuse_img)

        return rec_img2, rec_img2, com_img, uni_img1, uni_img2, fuse_img
