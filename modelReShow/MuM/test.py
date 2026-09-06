# ============================================================
# MuM 双模型对比测试脚本
# 同时运行 vit_small（自训练） & vit_large（预训练），对比重建效果
# ============================================================

import os, sys, torch, glob, functools
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mum.model import vit_small, mum_vitl16_decoderb, vit_base, vit_large
from mum.utils import transform_image
from mum.utils.viz import qualitative_evaluation

# ============================================================
# 配置区
# ============================================================
MASK_RATIO = 0.75
IMG_SIZE = 1024
VISIBLE = True
DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# vit_small 自训练 checkpoint
CKPT_SMALL = "/data1/Taohy/ModelReshow/modelReShow/MuM/checkpoints_traveluav/checkpoint-last.pth"

# vit_large 使用预训练权重（mum_vitl16_decoderb 内置下载）
USE_PRETRAINED_LARGE = True  # True=预训练, False=自训练 checkpoint
CKPT_LARGE = None  # 仅当 USE_PRETRAINED_LARGE=False 时使用

# ============================================================
# 测试图片来源
#   USE_VAL_SAMPLE = True  -> 从验证集 (SPLIT=val) 随机采样一组图片 (3帧)
#   USE_VAL_SAMPLE = False -> 使用下方 img_paths 指定的图片
# ============================================================
USE_VAL_SAMPLE = False
VAL_SEED = 42  # 采样随机种子 (固定可复现)
NUM_FRAMES = 3
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

# 使用 loss 最低的图片 (scene: 5beb6e66, rank 1, loss=0.163) [仅 USE_VAL_SAMPLE=False 时用]
img_paths = [
    "/data2/student1_ly/datasets/OpenUAV-QA/TravelUAV_dataset/TravelUAV_Train_15086/00070.png",
    "/data2/student1_ly/datasets/OpenUAV-QA/TravelUAV_dataset/TravelUAV_Train_15086/00071.png",
    "/data2/student1_ly/datasets/OpenUAV-QA/TravelUAV_dataset/TravelUAV_Train_15086/00072.png",
]
OUT_DIR = os.path.join(os.path.dirname(__file__), "comparison")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. 加载图片（共用）: 优先从验证集采样, 否则用 img_paths
# ============================================================
print(f"设备: {DEVICE}  |  mask_ratio: {MASK_RATIO}")

if USE_VAL_SAMPLE:
    from omegaconf import OmegaConf
    from mum.data.traveluav import TravelUAVDataset

    cfg = OmegaConf.load(CONFIG_PATH)
    # 解析数据集字符串: "TravelUAVDataset:DATA_DIR=...:SPLIT=...:SPLIT_RATIO=..."
    ds_str = cfg.data.datasets[0]
    ds_kwargs = {}
    for part in ds_str.split(":")[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            ds_kwargs[k.strip().upper()] = v.strip()
    ds_kwargs["SPLIT"] = "val"

    np.random.seed(VAL_SEED)  # 固定采样结果, 便于复现
    ds = TravelUAVDataset(
        data_dir=ds_kwargs["DATA_DIR"],
        common_config=cfg.data.common_config,
        split=ds_kwargs["SPLIT"],
        split_ratio=ds_kwargs.get("SPLIT_RATIO", 0.95),
    )
    sample = ds[0]
    raw_imgs = sample["images"]  # [S, 3, H, W] (config 分辨率, 如 256)
    imgs = (
        torch.nn.functional.interpolate(
            raw_imgs, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
        )
        .unsqueeze(0)
        .to(DEVICE)
    )  # [1, S, 3, IMG_SIZE, IMG_SIZE]
    print(
        f"从验证集采样: scene_id={sample['id']}  原始分辨率: {list(raw_imgs.shape[2:])} -> resize 到 {IMG_SIZE}"
    )

    # 把采样到的原始帧保存下来, 方便查看是哪些图
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    from torchvision.utils import save_image

    x = (imgs[0].cpu() * std + mean).clamp(0, 1)
    for i in range(x.shape[0]):
        p = os.path.join(OUT_DIR, f"val_sample_{i}.jpg")
        save_image(x[i], p)
        print(f"  frame{i}: {p}")
else:
    print(f"图片:")
    for p in img_paths:
        assert os.path.exists(p), f"图片不存在: {p}"
        print(f"  {p}")
    imgs = (
        torch.stack([transform_image(p, size=(IMG_SIZE, IMG_SIZE)) for p in img_paths])
        .unsqueeze(0)
        .to(DEVICE)
    )  # [1, S, 3, H, W]
print(f"输入形状: {imgs.shape}")


# ============================================================
# 2. 辅助函数：加载自训练 checkpoint
# ============================================================
def map_key(k: str) -> str:
    k = k.replace("module.", "")
    k = k.replace("encoder.", "")
    if "decoder.rope_embed" in k:
        k = k.replace("decoder.rope_embed", "rope_embed_decoder")
    elif k.startswith("decoder."):
        k = k.replace("decoder.", "")
    return k


def load_ckpt(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    elif "model" in sd:
        sd = sd["model"]
    new_sd = {map_key(k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"  ⚠ 缺失 key ({len(missing)}个)")
    if unexpected:
        print(f"  ⚠ 多余 key ({len(unexpected)}个)")
    if not missing and not unexpected:
        print("  ✅ 所有权重完全匹配!")


# ============================================================
# 3. 构建两个模型
# ============================================================
models = {}

# --- vit_small（自训练） ---
print("\n" + "=" * 60)
print("[vit_small] 构建模型...")
model_small = vit_base(patch_size=16, img_size=IMG_SIZE, norm_pix_loss=True)
load_ckpt(model_small, CKPT_SMALL)
model_small.to(DEVICE).eval()
n_small = sum(p.numel() for p in model_small.parameters()) / 1e6
print(f"  参数: {n_small:.1f}M  |  checkpoint: {os.path.basename(CKPT_SMALL)}")
models["vit_small (自训练)"] = model_small

# --- vit_large（预训练） ---
print("\n[vit_large] 构建模型...")
if USE_PRETRAINED_LARGE:
    print("  使用预训练权重 (mum_vitl16_decoderb)...")
    model_large = mum_vitl16_decoderb(
        pretrained=True,
        weights_path="/data1/Taohy/ModelReshow/modelReShow/MuM/MuM_ViTLarge_BaseDecoder.pth",
        img_size=IMG_SIZE,
        norm_pix_loss=True,
    )
else:
    assert CKPT_LARGE and os.path.exists(CKPT_LARGE), "请设置 CKPT_LARGE"
    model_large = vit_small.__class__  # never reached
model_large.to(DEVICE).eval()
n_large = sum(p.numel() for p in model_large.parameters()) / 1e6
print(f"  参数: {n_large:.1f}M  |  权重: 预训练 (github.com/davnords/mum)")
models["vit_large (预训练)"] = model_large

# ============================================================
# 4. 双模型推理 + 对比
# ============================================================
results = {}

for name, model in models.items():
    print(f"\n{'=' * 60}")
    print(f"[{name}] 推理中...")

    # 可视化
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    out_path = os.path.join(OUT_DIR, f"recon_{safe_name}.png")
    original_forward = model.forward
    model.forward = functools.partial(original_forward, mask_ratio=MASK_RATIO)
    qualitative_evaluation(model, imgs, path=out_path, visible=VISIBLE)
    model.forward = original_forward

    # 算 loss
    with torch.inference_mode():
        loss, pred, mask = model(imgs, mask_ratio=MASK_RATIO)
        print(f"  Loss (MSE, norm_pix): {loss.item():.6f}")
        print(f"  实际 mask 比例: {mask.float().mean().item():.3f}")
        print(f"  可视化: {out_path}")
        results[name] = {
            "loss": loss.item(),
            "mask_ratio": mask.float().mean().item(),
            "output": out_path,
        }

# ============================================================
# 5. 对比总结
# ============================================================
print(f"\n{'=' * 60}")
print("对比总结:")
print(f"{'模型':<25s} {'参数':>10s} {'Loss':>10s} {'Mask':>8s}")
print("-" * 55)
for name in models:
    n = sum(p.numel() for p in models[name].parameters()) / 1e6
    r = results[name]
    print(f"{name:<25s} {n:>8.1f}M {r['loss']:>10.6f} {r['mask_ratio']:>8.3f}")

names = list(results.keys())
delta = abs(results[names[0]]["loss"] - results[names[1]]["loss"])
if results[names[0]]["loss"] < results[names[1]]["loss"]:
    winner = names[0]
else:
    winner = names[1]
print(f"\n✅ Loss 更低: {winner} (差值 {delta:.6f})")
print(f"输出目录: {OUT_DIR}")
