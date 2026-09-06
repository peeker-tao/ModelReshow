# ============================================================
# 从 TravelUAV 验证集 (SPLIT=val) 找效果最好 / 最差的 3 帧序列
# 效果指标: 掩码重建 loss (越小越好, 与训练同口径)
# ============================================================

import os, sys, torch, json, time
import numpy as np
import functools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omegaconf import OmegaConf
from mum.model import vit_base
from mum.utils import transform_image
from mum.utils.viz import qualitative_evaluation
from mum.data.traveluav import TravelUAVDataset

# ============================================================
# 配置区
# ============================================================
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
CKPT_PATH   = "/data1/Taohy/ModelReshow/modelReShow/MuM/checkpoints_traveluav/checkpoint-last.pth"
MASK_RATIO  = 0.75
IMG_SIZE    = 224
NUM_FRAMES  = 3        # 每个序列的帧数
STRIDE      = 3        # 滑窗步长 (1=每个连续3帧都测, 最全但慢; 3=每隔3帧测一组)
MAX_SCENES  = None     # 限制验证集场景数, None=全部 (767 个)
TOP_N       = 3        # 输出最好/最差各 N 组

OUT_DIR     = os.path.join(os.path.dirname(__file__), "comparison", "best_worst")
JSON_PATH   = os.path.join(OUT_DIR, "best_worst_results.json")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. 构建验证集 (拿到所有场景的帧列表, 用滑窗遍历连续 3 帧)
# ============================================================
cfg = OmegaConf.load(CONFIG_PATH)
ds_str = cfg.data.datasets[0]
ds_kwargs = {}
for part in ds_str.split(":")[1:]:
    if "=" in part:
        k, v = part.split("=", 1)
        ds_kwargs[k.strip().upper()] = v.strip()
ds_kwargs["SPLIT"] = "val"

ds = TravelUAVDataset(
    data_dir=ds_kwargs["DATA_DIR"],
    common_config=cfg.data.common_config,
    split=ds_kwargs["SPLIT"],
    split_ratio=ds_kwargs.get("SPLIT_RATIO", 0.95),
)
scenes   = ds.scenes
frame_ls = ds.frames
if MAX_SCENES:
    scenes, frame_ls = scenes[:MAX_SCENES], frame_ls[:MAX_SCENES]
print(f"待扫描: {len(scenes)} 个验证集场景")

# ============================================================
# 2. 加载模型 (vit_base)
# ============================================================
model = vit_base(patch_size=16, img_size=IMG_SIZE, norm_pix_loss=True)

sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
if "model_state_dict" in sd:
    sd = sd["model_state_dict"]
elif "model" in sd:
    sd = sd["model"]

def map_key(k: str) -> str:
    k = k.replace("module.", "").replace("encoder.", "")
    if "decoder.rope_embed" in k:
        k = k.replace("decoder.rope_embed", "rope_embed_decoder")
    elif k.startswith("decoder."):
        k = k.replace("decoder.", "")
    return k

new_sd = {map_key(k): v for k, v in sd.items()}
missing, unexpected = model.load_state_dict(new_sd, strict=False)
if missing:
    print(f"  ⚠ 缺失 key ({len(missing)}个): {missing[:3]}...")
if unexpected:
    print(f"  ⚠ 多余 key ({len(unexpected)}个): {unexpected[:3]}...")
if not missing and not unexpected:
    print("  ✅ 所有权重完全匹配!")
model.to(DEVICE).eval()
print(f"模型: vit_base  |  Checkpoint: {os.path.basename(CKPT_PATH)}  |  mask_ratio={MASK_RATIO}")

# ============================================================
# 3. 扫描所有连续 3 帧序列, 计算重建 loss
# ============================================================
results = []   # (loss, [3帧路径], scene)
n_seqs  = 0
t0 = time.time()

for si, (scene, frames) in enumerate(zip(scenes, frame_ls)):
    n = len(frames)
    if n < NUM_FRAMES:
        continue
    for i in range(0, n - NUM_FRAMES + 1, STRIDE):
        p3 = [str(frames[i + k]) for k in range(NUM_FRAMES)]
        try:
            imgs = torch.stack(
                [transform_image(p, size=(IMG_SIZE, IMG_SIZE)) for p in p3]
            ).unsqueeze(0).to(DEVICE)  # [1, S, 3, H, W]
            with torch.inference_mode():
                loss, _, _ = model(imgs, mask_ratio=MASK_RATIO)
            results.append((float(loss.item()), p3, scene))
            n_seqs += 1
        except Exception:
            pass

    if (si + 1) % 10 == 0 or si == len(scenes) - 1:
        elapsed = time.time() - t0
        eta = elapsed / (si + 1) * len(scenes) - elapsed
        pct = (si + 1) / len(scenes) * 100
        bar_len = 30
        filled = int(bar_len * (si + 1) / len(scenes))
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  [{bar}] {pct:5.1f}%  [{si+1}/{len(scenes)}] 已跑 {n_seqs} 组  |  "
              f"耗时 {elapsed:.0f}s  |  剩余 ~{eta:.0f}s", end="", flush=True)

results.sort(key=lambda x: x[0])
total_time = time.time() - t0
print(f"\n总序列数: {n_seqs}  |  总耗时: {total_time:.0f}s")

best  = results[:TOP_N]
worst = results[-TOP_N:]

def dump_group(group, tag):
    print(f"\n{'='*60}")
    print(f"{tag} (loss 最小={group[0][0]:.6f}, 最大={group[-1][0]:.6f}):")
    for rank, (loss_val, p3, scene) in enumerate(group):
        print(f"  {tag} #{rank+1}: loss={loss_val:.6f}  |  scene={scene}")
        for p in p3:
            print(f"      {os.path.basename(p)}")

dump_group(best, "🏆 效果最好")
dump_group(worst, "💀 效果最差")

# ============================================================
# 4. 对最好/最差的样本做可视化 (Original / Masked / Predicted)
# ============================================================
def save_viz(group, tag, model):
    for rank, (loss_val, p3, scene) in enumerate(group):
        imgs = torch.stack(
            [transform_image(p, size=(IMG_SIZE, IMG_SIZE)) for p in p3]
        ).unsqueeze(0).to(DEVICE)
        safe = f"{tag}_rank{rank+1}_loss{loss_val:.4f}"
        out_path = os.path.join(OUT_DIR, f"recon_{safe}.png")
        orig_forward = model.forward
        model.forward = functools.partial(orig_forward, mask_ratio=MASK_RATIO)
        qualitative_evaluation(model, imgs, path=out_path, visible=True)
        model.forward = orig_forward
        print(f"  可视化 → {out_path}")

print("\n生成可视化...")
save_viz(best, "best", model)
save_viz(worst, "worst", model)

# ============================================================
# 5. 保存结果 JSON
# ============================================================
with open(JSON_PATH, "w") as f:
    json.dump({
        "config": {"ckpt": CKPT_PATH, "mask_ratio": MASK_RATIO,
                   "stride": STRIDE, "max_scenes": MAX_SCENES},
        "total_sequences": n_seqs,
        "total_time_s": total_time,
        "best": [{"rank": i+1, "loss": r[0], "scene": r[2], "images": r[1]}
                 for i, r in enumerate(best)],
        "worst": [{"rank": i+1, "loss": r[0], "scene": r[2], "images": r[1]}
                  for i, r in enumerate(worst)],
    }, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存到: {JSON_PATH}")
