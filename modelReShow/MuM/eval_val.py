# ============================================================
# MuM 验证集评估脚本 (TravelUAV val split)
# 用法:
#   python eval_val.py --ckpt checkpoints_traveluav/checkpoint-last.pth
#   python eval_val.py --ckpt .../checkpoint-step18000-xxx.pth --num-batches 500
# 说明:
#   - 在 val 集 (SPLIT=val) 上计算掩码重建 loss, 与训练 loss 同口径
#   - train loss vs val loss 对比: 差距小 = 泛化好; val 远高于 train = 过拟合
# ============================================================

import os
import sys
import argparse
import time

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mum.model import build_model, dtype_dict
from mum.data import DataAugmentationMAE, build_dynamic_dataloader


def parse_args():
    p = argparse.ArgumentParser(description="MuM val reconstruction loss eval")
    p.add_argument("--ckpt", type=str, required=True,
                   help="checkpoint 路径, 如 checkpoints_traveluav/checkpoint-last.pth")
    p.add_argument("--config", type=str, default="config.yaml",
                   help="训练配置 (用于模型结构与数据路径)")
    p.add_argument("--num-batches", type=int, default=200,
                   help="评估多少个 batch (每个 batch 16 样本, 200 个 ≈ 3200 样本)")
    p.add_argument("--batch-per-gpu", type=int, default=16,
                   help="每卡 batch (3 帧时建议 16)")
    p.add_argument("--mask-ratio", type=float, default=0.75,
                   help="掩码比例, 与训练一致 (默认取 config 里的值)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1. 加载配置
    cfg = OmegaConf.load(args.config)
    assert torch.cuda.is_available(), "需要 GPU"
    device = "cuda"

    # 2. 构建模型并加载权重
    model = build_model(cfg).to(device).eval()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd, strict=False)
    print(f"已加载 checkpoint: {args.ckpt}")

    # 3. 构建 val 数据加载器 (SPLIT=val)
    #    替换 config 里的数据集为 val split
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

    mask_ratio = args.mask_ratio if args.mask_ratio is not None else cfg.loss.mask_ratio
    print(f"评估: {args.num_batches} batches, mask_ratio={mask_ratio}, "
          f"batch={args.batch_per_gpu}, 精度={cfg.dtype}")

    # 4. 计算平均重建 loss
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

    mean_loss = float(np.mean(losses))
    elapsed = time.time() - t0
    print(f"\n========== 评估结果 ==========")
    print(f"样本数:        {total}")
    print(f"平均重建 loss: {mean_loss:.4f}")
    print(f"评估耗时:      {elapsed:.1f}s")
    print(f"==============================")

    # 5. 提示与训练 loss 对比
    print(f"\n提示: 把该值与你训练日志中的 loss 对比")
    print(f"  - val loss ≈ train loss  → 泛化良好")
    print(f"  - val loss >> train loss → 可能过拟合 (考虑增大数据/正则)")


if __name__ == "__main__":
    main()
