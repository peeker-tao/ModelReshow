# ============================================================
# 用预训练 MuM ViT-Large 权重在验证集 (SPLIT=val) 上评估重建 loss
# 可选: 同时评估自训练 vit_base, 输出对比
#
# 用法:
#   python eval_val_pretrained.py                          # 只跑 vit_large (预训练)
#   python eval_val_pretrained.py --compare                # 同时跑 vit_large + vit_base
#   python eval_val_pretrained.py --num-batches 500        # 跑更多批次
# ============================================================

import os
import sys
import argparse
import time

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mum.model import mum_vitl16_decoderb, vit_base, dtype_dict,vit_large
from mum.data import DataAugmentationMAE, build_dynamic_dataloader


def parse_args():
    p = argparse.ArgumentParser(description="MuM pretrained ViT-Large val loss eval")
    p.add_argument("--weights-large", type=str,
                   default="/data1/Taohy/ModelReshow/modelReShow/MuM/MuM_ViTLarge_BaseDecoder.pth",
                   help="vit_large 预训练权重路径")
    p.add_argument("--ckpt-base", type=str,
                   default="/data1/Taohy/ModelReshow/modelReShow/MuM/checkpoints_traveluav_256/checkpoint-vit_large-20260820_164658-step12000-loss0.2914-valloss0.3575.pth",
                   help="vit_base 自训练 checkpoint (--compare 时使用)")
    p.add_argument("--compare", action="store_true",
                   help="同时评估自训练 vit_base 并对比")
    p.add_argument("--config", type=str, default="config.yaml",
                   help="训练配置 (用于数据路径/模型结构)")
    p.add_argument("--num-batches", type=int, default=200,
                   help="评估多少个 batch (每个 batch 含 batch_per_gpu 个样本)")
    p.add_argument("--batch-per-gpu", type=int, default=8,
                   help="每卡 batch (vit_large 显存大, 默认 8; vit_base 可用 16)")
    p.add_argument("--mask-ratio", type=float, default=0.75,
                   help="掩码比例, 与训练一致 (默认取 config 里的值)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def build_val_dataloader(cfg, args, device):
    """构建 val 数据加载器 (SPLIT=val), 与训练同口径."""
    val_datasets = []
    for ds_str in cfg.data.datasets:
        parts = ds_str.split(":")
        new_parts = [p for p in parts if not p.upper().startswith("SPLIT")]
        new_parts.append("SPLIT=val")
        val_datasets.append(":".join(new_parts))
    print(f"val 数据集: {val_datasets}")

    image_aug = DataAugmentationMAE(img_size=cfg.train.img_size)
    dl = build_dynamic_dataloader(
        datasets=val_datasets,
        common_config=cfg.data.common_config,
        image_aug=image_aug,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=False,
        max_img_per_gpu=args.batch_per_gpu * 3,  # ×3 帧
    )
    return dl


def evaluate(model, dl, cfg, args, device):
    """在 val dataloader 上计算平均重建 loss, 返回 (mean_loss, num_samples, elapsed)."""
    mask_ratio = args.mask_ratio if args.mask_ratio is not None else cfg.loss.mask_ratio
    losses = []
    total = 0
    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= args.num_batches:
                break
            samples = batch["images"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=dtype_dict[cfg.dtype]):
                loss, _, _ = model(samples, mask_ratio=mask_ratio)
            losses.append(loss.item())
            total += samples.shape[0]
    elapsed = time.time() - t0
    return float(np.mean(losses)), total, elapsed


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = OmegaConf.load(args.config)
    assert torch.cuda.is_available(), "需要 GPU"
    device = "cuda"

    mask_ratio = args.mask_ratio if args.mask_ratio is not None else cfg.loss.mask_ratio
    print(f"评估配置: {args.num_batches} batches × {args.batch_per_gpu} 样本  |  "
          f"mask_ratio={mask_ratio}  |  精度={cfg.dtype}")
    print("=" * 60)

    dl = build_val_dataloader(cfg, args, device)
    results = {}

    # ---------- vit_large (预训练) ----------
    print("\n[vit_large] 加载预训练权重...")
    model_large = mum_vitl16_decoderb(
        pretrained=True,
        weights_path=args.weights_large,
        patch_size=16,
        img_size=cfg.train.img_size,
        norm_pix_loss=cfg.model.norm_pix_loss,
    ).to(device).eval()
    n_large = sum(p.numel() for p in model_large.parameters()) / 1e6
    print(f"  vit_large | 参数: {n_large:.1f}M | 权重: {os.path.basename(args.weights_large)}")
    mean_large, total, elapsed = evaluate(model_large, dl, cfg, args, device)
    print(f"  平均重建 loss: {mean_large:.4f}  |  样本: {total}  |  耗时: {elapsed:.1f}s")
    results["vit_large (预训练)"] = mean_large
    del model_large
    torch.cuda.empty_cache()

    # ---------- vit_base (自训练, 可选) ----------
    if args.compare:
        print("\n[vit_base] 加载自训练 checkpoint...")
        assert os.path.exists(args.ckpt_base), f"checkpoint 不存在: {args.ckpt_base}"
        model_base = vit_base(patch_size=16, img_size=cfg.train.img_size,
                              norm_pix_loss=cfg.model.norm_pix_loss)
        sd = torch.load(args.ckpt_base, map_location="cpu", weights_only=True)
        if "model" in sd:
            sd = sd["model"]

        def map_key(k: str) -> str:
            k = k.replace("module.", "").replace("encoder.", "")
            if "decoder.rope_embed" in k:
                k = k.replace("decoder.rope_embed", "rope_embed_decoder")
            elif k.startswith("decoder."):
                k = k.replace("decoder.", "")
            return k

        new_sd = {map_key(k): v for k, v in sd.items()}
        missing, unexpected = model_base.load_state_dict(new_sd, strict=False)
        if missing:
            print(f"  ⚠ 缺失 key ({len(missing)}个)")
        if unexpected:
            print(f"  ⚠ 多余 key ({len(unexpected)}个)")
        if not missing and not unexpected:
            print("  ✅ 所有权重完全匹配!")
        model_base.to(device).eval()
        n_base = sum(p.numel() for p in model_base.parameters()) / 1e6
        print(f"  vit_base | 参数: {n_base:.1f}M | ckpt: {os.path.basename(args.ckpt_base)}")
        mean_base, total, elapsed = evaluate(model_base, dl, cfg, args, device)
        print(f"  平均重建 loss: {mean_base:.4f}  |  样本: {total}  |  耗时: {elapsed:.1f}s")
        results["vit_base (自训练)"] = mean_base

    # ---------- 汇总 ----------
    print("\n" + "=" * 60)
    print("评估结果汇总 (mask_ratio={:.2f}, {} 样本):".format(mask_ratio, total))
    for name, loss in results.items():
        print(f"  {name:<22s}  loss = {loss:.4f}")
    if len(results) == 2:
        a, b = list(results.values())
        diff = b - a  # vit_base - vit_large
        print(f"\n  差异 (vit_base - vit_large): {diff:+.4f}"
              f"  -> {'vit_base 更好' if diff < 0 else 'vit_large 更好'}  (loss 越低越好)")
    print("=" * 60)


if __name__ == "__main__":
    main()
