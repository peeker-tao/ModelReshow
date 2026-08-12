# ============================================================
# MuM 双模型对比测试脚本
# 同时运行 vit_small（自训练） & vit_large（预训练），对比重建效果
# ============================================================

import os, sys, torch, glob, functools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mum.model import vit_small, mum_vitl16_decoderb, vit_base
from mum.utils import transform_image
from mum.utils.viz import qualitative_evaluation

# ============================================================
# 配置区
# ============================================================
MASK_RATIO = 0.75
IMG_SIZE = 224
VISIBLE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# vit_small 自训练 checkpoint
CKPT_SMALL = "modelReShow/mum/checkpoints/mum_0730-104027_step34200.pth"

# vit_large 使用预训练权重（mum_vitl16_decoderb 内置下载）
USE_PRETRAINED_LARGE = True  # True=预训练, False=自训练 checkpoint
CKPT_LARGE = None  # 仅当 USE_PRETRAINED_LARGE=False 时使用

# 使用 loss 最低的图片 (scene: 5beb6e66, rank 1, loss=0.163)
img_paths = [
    "/data/data_taohy/datasets/BlendedMVS/57f8d9bbe73f6760f10e916a/blended_images/00000000.jpg",
    "/data/data_taohy/datasets/BlendedMVS/57f8d9bbe73f6760f10e916a/blended_images/00000001.jpg",
    "/data/data_taohy/datasets/BlendedMVS/57f8d9bbe73f6760f10e916a/blended_images/00000002.jpg",
]
OUT_DIR = os.path.join(os.path.dirname(__file__), "comparison")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. 加载图片（共用）
# ============================================================
print(f"设备: {DEVICE}  |  mask_ratio: {MASK_RATIO}")
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
    else:
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
        pretrained=True, img_size=IMG_SIZE, norm_pix_loss=True
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
