# ============================================================
# 遍历 BlendedMVS 所有图片，找出 loss 最低的 3 帧序列
# ============================================================

import os, sys, torch, glob, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mum.model import vit_small, vit_base
from mum.utils import transform_image

# ============================================================
# 配置区
# ============================================================
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH   = "mum_0730-133650_final_step15000.pth"
MASK_RATIO  = 0.75
IMG_SIZE    = 224
IMG_DIR     = "/data/data_taohy/datasets/BlendedMVS"
TOP_N       = 30
STRIDE      = 3
MAX_SCENES  = None
OUTPUT_FILE = "/data/data_taohy/modelReShow/MuM/best_images_results.json"

# ============================================================
# 1. 加载模型
# ============================================================
ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mum", "checkpoints"))
ckpt_path = CKPT_PATH if os.path.isabs(CKPT_PATH) else os.path.join(ckpt_dir, CKPT_PATH)

ckpt_size = os.path.getsize(ckpt_path)
if ckpt_size < 500 * 1024 * 1024:
    model = vit_small(patch_size=16, img_size=IMG_SIZE, norm_pix_loss=True)
    model_type = "vit_small"
else:
    model = vit_base(patch_size=16, img_size=IMG_SIZE, norm_pix_loss=True)
    model_type = "vit_base"

sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
if "model_state_dict" in sd:
    sd = sd["model_state_dict"]

def map_key(k):
    k = k.replace("module.", "").replace("encoder.", "")
    if "decoder.rope_embed" in k:
        k = k.replace("decoder.rope_embed", "rope_embed_decoder")
    else:
        k = k.replace("decoder.", "")
    return k

model.load_state_dict({map_key(k): v for k, v in sd.items()}, strict=False)
model.to(DEVICE).eval()
print(f"模型: {model_type}  |  Checkpoint: {os.path.basename(ckpt_path)}  |  mask_ratio={MASK_RATIO}")

# ============================================================
# 2. 扫描
# ============================================================
scene_dirs = sorted(glob.glob(f"{IMG_DIR}/*/blended_images"))
if MAX_SCENES:
    scene_dirs = scene_dirs[:MAX_SCENES]

results = []
n_seqs = 0
n_scenes = len(scene_dirs)

t0 = time.time()
for si, scene_dir in enumerate(scene_dirs):
    scene = os.path.basename(os.path.dirname(scene_dir))
    paths = sorted([p for p in glob.glob(f"{scene_dir}/*.jpg") if "_masked" not in p])
    n = len(paths)
    if n < 3:
        continue
    for i in range(0, n - 2, STRIDE):
        n_seqs += 1
        p3 = [paths[i], paths[i+1], paths[i+2]]
        try:
            imgs = torch.stack(
                [transform_image(p, size=(IMG_SIZE, IMG_SIZE)) for p in p3]
            ).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                loss, _, _ = model(imgs, mask_ratio=MASK_RATIO)
            results.append((float(loss.item()), p3, scene))
        except Exception as e:
            pass
    elapsed = time.time() - t0
    eta = elapsed / (si + 1) * n_scenes - elapsed
    print(f"  [{si+1}/{n_scenes}] {scene} ({n}帧)  |  已跑 {n_seqs} 组  |  耗时 {elapsed:.0f}s  |  剩余 ~{eta:.0f}s", flush=True)

# ============================================================
# 3. 排序 & 保存
# ============================================================
results.sort(key=lambda x: x[0])

total_time = time.time() - t0
print(f"\n总序列数: {n_seqs}  |  总耗时: {total_time:.0f}s")
print(f"\n{'='*60}")
print(f"🏆 Loss 最低的 {TOP_N} 组:")
print(f"{'='*60}")

for rank, (loss_val, paths, scene) in enumerate(results[:TOP_N]):
    print(f"\n{rank+1:3d}. loss={loss_val:.6f}  |  {scene}")
    for p in paths:
        print(f"     {os.path.basename(p):>20s}  ->  {p}")

# 保存 JSON
with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "config": {"ckpt": CKPT_PATH, "mask_ratio": MASK_RATIO},
        "total_sequences": n_seqs,
        "total_time_s": total_time,
        "top_results": [{"rank": i+1, "loss": r[0], "scene": r[2], "images": r[1]} for i, r in enumerate(results[:TOP_N])]
    }, f, indent=2)
print(f"\n结果已保存到: {OUTPUT_FILE}")
