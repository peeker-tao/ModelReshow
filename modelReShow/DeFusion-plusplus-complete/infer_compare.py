"""指定图片（base 名）→ 两个权重分别推理 → 生成融合对比图。

用法示例:
    # 默认：对比旧模型(无教师) vs 新模型(FLIR 教师)，指定 2 组图片
    python infer_compare.py --base 00004N 00008N

    # 自定义两个权重和标签
    python infer_compare.py --base 00004N \
        --weights pretrained/msrs_mcud_step20000.pth pretrained/msrs_mcud_flir_step20000.pth \
        --labels "旧模型(无教师)" "新模型(FLIR教师)"

    # 直接用 IR/VI 图片路径（不查 MSRS 数据集）
    python infer_compare.py --ir /path/ir.png --vi /path/vi.png --name 00004N
"""
import argparse
import os

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToPILImage

from models.MUCMIModelTest import MUCMIMNetTest
from data.visir_fusion_dataset import TestDataset as IVFTestDataset

# MSRS 数据集配置（与 option/test/*.yaml 一致）
DATAROOT = '/data1/Taohy/dataset/MSRS'
INFRARE_NAME = 'test/ir'
VISIBLE_NAME = 'test/vi'

# 默认对比的两个权重
DEFAULT_WEIGHTS = [
    '/data1/Taohy/ModelReshow/modelReShow/DeFusion-plusplus-complete/experiments/COCO_MSRS_MCUD_demo/models/checkpoint-Defusion-plusplus-20260905_130348-step5000-loss36.4892-valloss27.0931.pth',       # 旧：无教师
    '/data1/Taohy/ModelReshow/modelReShow/DeFusion-plusplus-complete/experiments/COCO_MSRS_MCUD_demo/models/checkpoint-last.pth',  # 新：FLIR 教师
]
DEFAULT_LABELS = ['旧模型(无教师)', '新模型(FLIR教师)']

# 6 张图后缀与含义
NAMES = ['over', 'under', 'upper', 'lower', 'common', 'recover']
CN = {
    'over':   'IR 红外输入',
    'under':  'VI 可见光输入',
    'upper':  '独有特征1',
    'lower':  '独有特征2',
    'common': '共有特征',
    'recover': '融合结果',
}
PAD = 16  # 与 test.py 一致：推理前 reflect pad，推理后裁掉

# 中文标签字体（项目内下载的文泉驿微米黑），找不到则回退 DejaVu
FONT_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts', 'wqy-microhei.ttc'),
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
]


def make_font(size):
    for fp in FONT_CANDIDATES:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def load_model(weight_path, device):
    model = MUCMIMNetTest()
    state = torch.load(weight_path, map_location=device)
    # 兼容三种格式：{'state_dict': ...} / {'model': ...} / 裸 state_dict
    sd = state.get('state_dict', state.get('model', state))
    # 训练 checkpoint 是 MUCMIMNet 的超集（含教师/latent_predict/mim 等 898 keys），
    # 只保留测试模型 MUCMIMNetTest 真正需要的 352 keys。
    keep = set(model.state_dict().keys())
    clean_sd = {k: v for k, v in sd.items() if k in keep}
    if len(clean_sd) != len(sd):
        print(f'  [load_model] 过滤训练专用 keys: {len(sd)} -> {len(clean_sd)}')
    model.load_state_dict(clean_sd, strict=True)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def infer_pair(model, ir_img, vi_img, device):
    """输入 IR/VI 的 PIL Image，输出 6 张 [0,1] 的 (C,H,W) tensor。"""
    import torchvision.transforms as T
    t = T.ToTensor()
    o_img = t(ir_img).unsqueeze(0)   # IR
    u_img = t(vi_img).unsqueeze(0)   # VI

    o_img = F.pad(o_img, (PAD, PAD, PAD, PAD), mode='reflect').to(device)
    u_img = F.pad(u_img, (PAD, PAD, PAD, PAD), mode='reflect').to(device)

    common_part, upper_part, lower_part, fusion_part = model(o_img, u_img)

    def crop(x):
        return x[:, :, PAD:-PAD, PAD:-PAD]

    o_img = crop(o_img)
    u_img = crop(u_img)
    common_part = crop(common_part)
    upper_part = crop(upper_part)
    lower_part = crop(lower_part)
    fusion_part = crop(fusion_part)

    return {
        'over': o_img[0].cpu(),
        'under': u_img[0].cpu(),
        'upper': upper_part[0].cpu(),
        'lower': lower_part[0].cpu(),
        'common': common_part[0].cpu(),
        'recover': fusion_part[0].cpu(),
    }


def panel(imgs_dict, title, title_color):
    """把 6 张 tensor 拼成 3x2 网格 + 标题 + 标签，返回 PIL 面板。"""
    pil = {}
    for k in NAMES:
        pil[k] = ToPILImage()(imgs_dict[k].clamp(0, 1))

    w, h = pil[NAMES[0]].size
    cols, rows = 3, 2
    pad, gap = 12, 8
    label_h, title_h = 30, 46

    panel_w = cols * w + (cols - 1) * gap + 2 * pad
    panel_h = title_h + rows * (h + label_h) + (rows - 1) * gap + 2 * pad
    canvas = Image.new('RGB', (panel_w, panel_h), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)
    f_title = make_font(30)
    f_label = make_font(20)
    dr.text((pad, 8), title, fill=title_color, font=f_title)

    for i, k in enumerate(NAMES):
        r, c = divmod(i, cols)
        x = pad + c * (w + gap)
        yy = title_h + r * (h + label_h + gap)
        canvas.paste(pil[k], (x, yy))
        tw = dr.textlength(CN[k], font=f_label)
        dr.text((x + (w - tw) // 2, yy + h + 4), CN[k], fill=(0, 0, 0), font=f_label)
    return canvas


def find_msrs_pair(base):
    """从 MSRS test 集按 base 名返回 (ir_path, vi_path)。"""
    ir_dir = os.path.join(DATAROOT, INFRARE_NAME)
    vi_dir = os.path.join(DATAROOT, VISIBLE_NAME)
    for d in (ir_dir, vi_dir):
        for f in os.listdir(d):
            if f.split('.')[0] == base:
                pass  # 见下方：两个目录都要命中
    ir_files = {f.split('.')[0]: f for f in os.listdir(ir_dir)}
    vi_files = {f.split('.')[0]: f for f in os.listdir(vi_dir)}
    if base not in ir_files or base not in vi_files:
        raise FileNotFoundError(f'base "{base}" 未在 MSRS test 集中找到')
    return (os.path.join(ir_dir, ir_files[base]),
            os.path.join(vi_dir, vi_files[base]))


def load_image(path, is_ir):
    """复现 IVFTestDataset 的读取方式：IR 转灰度再转 RGB，VI 转 RGB。"""
    if is_ir:
        return Image.open(path).convert('L').convert('RGB')
    return Image.open(path).convert('RGB')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', nargs='*', default=[],
                   help='MSRS test 集中的 base 名（如 00004N），可多个')
    p.add_argument('--ir', default=None, help='直接指定 IR 图片路径')
    p.add_argument('--vi', default=None, help='直接指定 VI 图片路径')
    p.add_argument('--name', default=None, help='直接指定图片对时使用的名字')
    p.add_argument('--weights', nargs='*', default=DEFAULT_WEIGHTS,
                   help='两个权重路径')
    p.add_argument('--labels', nargs='*', default=DEFAULT_LABELS,
                   help='两个权重的显示标签')
    p.add_argument('--out', default='results', help='输出目录')
    args = p.parse_args()

    assert len(args.weights) == 2, '需要恰好两个权重（--weights a.pth b.pth）'
    if len(args.labels) != 2:
        args.labels = ['权重A', '权重B']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 收集 (name, ir_path, vi_path) 列表
    pairs = []
    if args.ir and args.vi:
        pairs.append((args.name or os.path.splitext(os.path.basename(args.vi))[0],
                      args.ir, args.vi))
    for base in args.base:
        ir_path, vi_path = find_msrs_pair(base)
        pairs.append((base, ir_path, vi_path))
    if not pairs:
        raise SystemExit('请用 --base 00004N 或 --ir/--vi 指定图片')

    # 加载两个模型
    models = [load_model(w, device) for w in args.weights]

    os.makedirs(args.out, exist_ok=True)
    for name, ir_path, vi_path in pairs:
        ir_img = load_image(ir_path, is_ir=True)
        vi_img = load_image(vi_path, is_ir=False)
        print(f'[{name}] IR={ir_path}  VI={vi_path}')

        # 每个模型单独输出一张 6 图面板（不上下堆叠）
        for mi, (model, label) in enumerate(zip(models, args.labels)):
            imgs = infer_pair(model, ir_img, vi_img, device)
            color = (20, 90, 180) if mi == 0 else (180, 20, 20)
            pn = panel(imgs, f'{label}  —  {name}', color)
            out = os.path.join(args.out, f'comparison_{name}_model{mi + 1}.png')
            pn.save(out)
            print(f'saved -> {out}  ({pn.size})')
        print()


if __name__ == '__main__':
    main()
