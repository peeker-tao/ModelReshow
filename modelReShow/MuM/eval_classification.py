# ============================================================
# MuM Encoder + MLP 分类评测脚本
# 冻结 MuM (ViT-L, MIM 预训练) 的 encoder, 提取特征,
# 训练 Linear / MLP 分类头, 在经典数据集上评测 top-1 准确率。
#
# 用法示例:
#   # 自训练 checkpoint, CIFAR-10, MLP 头
#   python eval_classification.py --dataset cifar10 --head mlp
#
#   # 官方预训练权重, CIFAR-100, 线性头
#   python eval_classification.py --weights pretrained --dataset cifar100 --head linear
#
#   # 本地已有 MNIST 数据, 快速验证
#   python eval_classification.py --dataset mnist --data-root /data1/Taohy/ModelReshow/minst手写数字识别 --epochs 2
# ============================================================

import os
import sys
import argparse
import time
import hashlib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mum.model import mum_vitl16_decoderb, vit_large, vit_base

# ------------------------------------------------------------
# 1. 参数
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="MuM Encoder + MLP 分类评测")
    p.add_argument("--weights", type=str, default="pretrained",
                   help="权重来源: pretrained=官方预训练权重; 或自训练 checkpoint 路径 (自动识别 vit_base/vit_large)")
    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["cifar10", "cifar100", "mnist", "fashionmnist", "svhn"],
                   help="经典分类数据集")
    p.add_argument("--data-root", type=str, default=None,
                   help="数据集根目录, 默认 <脚本目录>/data")
    p.add_argument("--img-size", type=int, default=256, help="送入 encoder 的分辨率, 与预训练一致")
    p.add_argument("--feat", type=str, default="cls", choices=["cls", "meanpool", "concat"],
                   help="特征类型: cls=CLS token, meanpool=patch均值, concat=两者拼接")
    p.add_argument("--head", type=str, default="mlp", choices=["linear", "mlp"],
                   help="分类头结构")
    p.add_argument("--hidden", type=int, default=1024, help="MLP 隐藏层维度")
    p.add_argument("--epochs", type=int, default=20, help="分类头训练轮数")
    p.add_argument("--lr", type=float, default=1e-3, help="分类头学习率")
    p.add_argument("--batch-size", type=int, default=512, help="分类头训练 batch")
    p.add_argument("--feat-batch", type=int, default=32, help="特征提取 batch")
    p.add_argument("--device", type=str, default="cuda:1", help="使用的设备")
    p.add_argument("--gpus", type=str, default=None,
                   help="逗号分隔多卡 id (如 0,1,2,3,4,5), 用 DataParallel 数据并行; 默认单卡 --device")
    p.add_argument("--local-rank", type=int, default=None,
                   help="torchrun 注入的本地 rank, 无需手动指定")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-cache", action="store_true", help="不缓存提取的特征")
    p.add_argument("--finetune", action="store_true", help="端到端微调 encoder (而非冻结特征+探针)")
    p.add_argument("--ft-epochs", type=int, default=10, help="微调轮数")
    p.add_argument("--ft-lr", type=float, default=1e-4, help="微调时 encoder 的学习率")
    p.add_argument("--ft-batch-size", type=int, default=16, help="微调 batch size")
    return p.parse_args()


# ------------------------------------------------------------
# 1.5 分布式 (DDP) 工具
# ------------------------------------------------------------
def setup_distributed():
    """torchrun 启动时初始化 DDP, 返回是否处于分布式模式."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        return True
    return False


def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0


# ------------------------------------------------------------
# 2. 数据集 & 预处理
# ------------------------------------------------------------
DATASETS = {
    "cifar10": (datasets.CIFAR10, 10),
    "cifar100": (datasets.CIFAR100, 100),
    "mnist": (datasets.MNIST, 10),
    "fashionmnist": (datasets.FashionMNIST, 10),
    "svhn": (datasets.SVHN, 10),
}


def build_transform(args, train=False):
    t = []
    if args.dataset in ("mnist", "fashionmnist"):
        t.append(transforms.Grayscale(num_output_channels=3))  # 单通道 -> RGB
    t += [
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(t)


def build_dataloaders(args):
    ds_cls, num_classes = DATASETS[args.dataset]
    root = args.data_root or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    if args.dataset == "svhn":
        train_ds = ds_cls(root, split="train", transform=build_transform(args), download=True)
        test_ds = ds_cls(root, split="test", transform=build_transform(args), download=True)
    else:
        train_ds = ds_cls(root, train=True, transform=build_transform(args), download=True)
        test_ds = ds_cls(root, train=False, transform=build_transform(args), download=True)

    train_loader = DataLoader(train_ds, batch_size=args.feat_batch, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.feat_batch, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, drop_last=False)
    return train_loader, test_loader, num_classes


# ------------------------------------------------------------
# 3. Encoder 加载 & 特征提取
# ------------------------------------------------------------
def _detect_embed_dim(weights_path):
    """从 checkpoint 推断 encoder 的 embed_dim (vit_base=768, vit_large=1024)."""
    ck = torch.load(weights_path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    if "cls_token" in sd:
        return sd["cls_token"].shape[-1]
    return sd["patch_embed.proj.weight"].shape[0]


def load_encoder(args, freeze=True):
    if args.weights == "pretrained":
        local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MuM_ViTLarge_BaseDecoder.pth")
        weights_path = local if os.path.exists(local) else None
        print(f"[encoder] 官方预训练权重: {weights_path or '下载中...'}")
    else:
        weights_path = args.weights
        assert os.path.exists(weights_path), f"checkpoint 不存在: {weights_path}"
        print(f"[encoder] 自训练权重: {weights_path}")

    # 根据 checkpoint 结构自动选择 vit_base / vit_large
    if weights_path and os.path.exists(weights_path):
        embed_dim = _detect_embed_dim(weights_path)
        arch = "vit_base" if embed_dim == 768 else "vit_large"
        print(f"[encoder] 检测到架构: {arch} (embed_dim={embed_dim})")
        if arch == "vit_base":
            model = vit_base(patch_size=16, img_size=args.img_size, norm_pix_loss=True)
            ck = torch.load(weights_path, map_location="cpu", weights_only=False)
            sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
            model.load_state_dict(sd, strict=True)
        else:
            model = mum_vitl16_decoderb(pretrained=True, weights_path=weights_path,
                                        patch_size=16, img_size=args.img_size, norm_pix_loss=True)
    else:
        model = mum_vitl16_decoderb(pretrained=True, weights_path=None,
                                    patch_size=16, img_size=args.img_size, norm_pix_loss=True)
    model = model.to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    if freeze:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)  # 冻结 encoder
        print(f"[encoder] ViT-L 参数: {n_params / 1e6:.1f}M (已冻结)")
    else:
        # finetune 分类只训练 encoder, 冻结 decoder (decoder 不参与 forward_features,
        # 解冻会导致 DDP 报 unused parameters)
        for name, p in model.named_parameters():
            if name.startswith("decoder_") or name == "mask_token":
                p.requires_grad_(False)
            else:
                p.requires_grad_(True)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[encoder] ViT-L 参数: {n_params / 1e6:.1f}M (可训练 {n_trainable / 1e6:.1f}M, 微调)")
    return model


class EncoderWrapper(nn.Module):
    """封装 encoder 的特征提取接口, 供单卡或 DataParallel 多卡统一调用.
    autocast 放在 forward 内部, 保证多卡时在 replica 线程里生效."""
    def __init__(self, encoder, feat="cls"):
        super().__init__()
        self.encoder = encoder
        self.feat = feat

    def forward(self, x):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.encoder.forward_features(x)
        cls_tok = out["x_norm_cls_token"].float()                  # [B, C]
        patch_tok = out["x_norm_patchtokens"].float().mean(dim=1)  # [B, C]
        if self.feat == "cls":
            return cls_tok
        elif self.feat == "meanpool":
            return patch_tok
        return torch.cat([cls_tok, patch_tok], dim=-1)


@torch.no_grad()
def extract_features(wrapper, loader, device):
    wrapper.eval()
    all_feats, all_labels = [], []
    t0 = time.time()
    for i, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        feats = wrapper(imgs)
        all_feats.append(feats.float().cpu())
        all_labels.append(labels.clone())
        if (i + 1) % 100 == 0:
            print(f"  提取特征 {i + 1}/{len(loader)}  ({time.time() - t0:.0f}s)")
    feats = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    print(f"[feature] 完成: {feats.shape} 耗时 {time.time() - t0:.0f}s")
    return feats, labels


# ------------------------------------------------------------
# 4. 分类头
# ------------------------------------------------------------
class LinearProbe(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class MLPProbe(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=1024, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_head(train_feats, train_labels, test_feats, test_labels,
               num_classes, args, in_dim):
    torch.manual_seed(args.seed)
    if args.head == "linear":
        model = LinearProbe(in_dim, num_classes)
    else:
        model = MLPProbe(in_dim, num_classes, hidden=args.hidden)

    model = model.to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    train_ds = TensorDataset(train_feats, train_labels)
    test_ds = TensorDataset(test_feats, test_labels)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    best_acc, best_epoch = 0.0, -1
    for epoch in range(args.epochs):
        model.train()
        total, correct, loss_sum, n_batch = 0, 0, 0.0, 0
        for feats, labels in train_loader:
            feats, labels = feats.to(args.device), labels.to(args.device)
            logits = model(feats)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            loss_sum += loss.item()
            n_batch += 1

        # 测试
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for feats, labels in test_loader:
                feats, labels = feats.to(args.device), labels.to(args.device)
                logits = model(feats)
                test_correct += (logits.argmax(dim=1) == labels).sum().item()
                test_total += labels.size(0)
        test_acc = test_correct / test_total
        if test_acc > best_acc:
            best_acc, best_epoch = test_acc, epoch

        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch + 1:3d} | train acc {correct / total:.4f} | "
                  f"loss {loss_sum / n_batch:.4f} | test acc {test_acc:.4f}")
    return best_acc, best_epoch


# ------------------------------------------------------------
# 4.5 端到端微调 (encoder + 分类头一起训练)
# ------------------------------------------------------------
class FinetuneModel(nn.Module):
    def __init__(self, encoder, num_classes, feat, hidden=1024, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.feat = feat
        in_dim = encoder.embed_dim * (2 if feat == "concat" else 1)
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, imgs):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.encoder.forward_features(imgs)
            cls_tok = out["x_norm_cls_token"]
            patch_tok = out["x_norm_patchtokens"]
            mean_pool = patch_tok.mean(dim=1)
            if self.feat == "cls":
                feats = cls_tok
            elif self.feat == "meanpool":
                feats = mean_pool
            else:
                feats = torch.cat([cls_tok, mean_pool], dim=-1)
            return self.head(feats)


def finetune_encoder(args, encoder):
    ds_cls, num_classes = DATASETS[args.dataset]
    root = args.data_root or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    if args.dataset == "svhn":
        train_ds = ds_cls(root, split="train", transform=build_transform(args), download=True)
        test_ds = ds_cls(root, split="test", transform=build_transform(args), download=True)
    else:
        train_ds = ds_cls(root, train=True, transform=build_transform(args), download=True)
        test_ds = ds_cls(root, train=False, transform=build_transform(args), download=True)

    use_ddp = dist.is_initialized()
    rank = dist.get_rank() if use_ddp else 0
    world_size = dist.get_world_size() if use_ddp else 1

    if use_ddp:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(train_ds, batch_size=args.ft_batch_size, sampler=train_sampler,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=args.ft_batch_size, sampler=test_sampler,
                                 num_workers=args.num_workers, pin_memory=True, drop_last=False)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.ft_batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=args.ft_batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model = FinetuneModel(encoder, num_classes, args.feat, hidden=args.hidden).to(args.device)
    if use_ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
        if rank == 0:
            print(f"[finetune] 使用 DDP 多卡: {world_size} 张卡, 每卡 batch={args.ft_batch_size}")

    # encoder 用较小 lr, 分类头用较大 lr (DDP 下取 module)
    base_model = model.module if hasattr(model, "module") else model
    encoder_params = [p for p in base_model.encoder.parameters()]
    head_params = [p for p in base_model.head.parameters()]
    opt = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.ft_lr},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.ft_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    best_acc, best_epoch = 0.0, -1
    for epoch in range(args.ft_epochs):
        if use_ddp:
            train_sampler.set_epoch(epoch)
        model.train()
        total, correct, loss_sum, n_batch = 0, 0, 0.0, 0
        for imgs, labels in train_loader:
            imgs = imgs.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            opt.zero_grad()
            logits = model(imgs)
            loss = F.cross_entropy(logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += labels.size(0)
            correct += (logits.float().argmax(dim=1) == labels).sum().item()
            loss_sum += loss.item()
            n_batch += 1

        # 测试
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(args.device, non_blocking=True)
                labels = labels.to(args.device, non_blocking=True)
                logits = model(imgs)
                test_correct += (logits.float().argmax(dim=1) == labels).sum().item()
                test_total += labels.size(0)
        if use_ddp:
            stats = torch.tensor([total, correct, loss_sum, n_batch,
                                  test_correct, test_total], dtype=torch.float64, device=args.device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total, correct = int(stats[0]), int(stats[1])
            loss_sum, n_batch = float(stats[2]), int(stats[3])
            test_correct, test_total = int(stats[4]), int(stats[5])
        test_acc = test_correct / test_total
        if test_acc > best_acc:
            best_acc, best_epoch = test_acc, epoch

        if rank == 0:
            print(f"  epoch {epoch + 1:3d} | train acc {correct / total:.4f} | "
                  f"loss {loss_sum / n_batch:.4f} | test acc {test_acc:.4f}")
    return best_acc, best_epoch


# ------------------------------------------------------------
# 5. 主流程
# ------------------------------------------------------------
def main():
    args = parse_args()
    use_ddp = setup_distributed()

    if use_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        args.device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[config] dataset={args.dataset} feat={args.feat} head={args.head} "
          f"epochs={args.epochs} lr={args.lr} device={args.device}")

    # 加载 encoder (微调模式下解冻)
    encoder = load_encoder(args, freeze=not args.finetune)

    # 端到端微调分支
    if args.finetune:
        if is_main_process():
            print(f"[finetune] 端到端微调 ViT-L encoder + {args.head} 头 "
                  f"({args.ft_epochs} epochs, enc_lr={args.ft_lr}, bs={args.ft_batch_size})")
        best_acc, best_epoch = finetune_encoder(args, encoder)
        if is_main_process():
            print("=" * 60)
            print(f"结果 | MuM({os.path.basename(args.weights)}) finetune + {args.head} "
                  f"[{args.feat}] @ {args.dataset}")
            print(f"Top-1 准确率: {best_acc * 100:.2f}% (best epoch {best_epoch + 1})")
            print("=" * 60)
        if use_ddp:
            dist.destroy_process_group()
        return

    # probe 分支在 DDP 下仅由主进程执行 (特征提取+训练头都很快, 单卡即可)
    if use_ddp and not is_main_process():
        dist.destroy_process_group()
        return

    # 构建数据
    train_loader, test_loader, num_classes = build_dataloaders(args)
    print(f"[data] {args.dataset}: train={len(train_loader.dataset)} "
          f"test={len(test_loader.dataset)} classes={num_classes}")

    # 提取特征 (带缓存, 缓存 key 用完整路径哈希, 防止不同 checkpoint 特征串用)
    weights_tag = hashlib.md5(args.weights.encode("utf-8")).hexdigest()[:12]
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                              f"feats_{args.dataset}_{args.feat}_{args.img_size}_{weights_tag}.pt")
    if not args.no_cache and os.path.exists(cache_path):
        print(f"[cache] 加载缓存特征: {cache_path}")
        saved = torch.load(cache_path, map_location="cpu", weights_only=True)
        train_feats, train_labels, test_feats, test_labels = (
            saved["train_feats"], saved["train_labels"], saved["test_feats"], saved["test_labels"])
    else:
        print("[feature] 提取训练集特征 ...")
        feat_extractor = EncoderWrapper(encoder, feat=args.feat)
        if args.gpus and not use_ddp:
            device_ids = [int(x) for x in args.gpus.split(",")]
            feat_extractor = nn.DataParallel(feat_extractor, device_ids=device_ids)
            print(f"[probe] 特征提取使用 DataParallel 多卡: {args.gpus}")
        train_feats, train_labels = extract_features(feat_extractor, train_loader, args.device)
        print("[feature] 提取测试集特征 ...")
        test_feats, test_labels = extract_features(feat_extractor, test_loader, args.device)
        if not args.no_cache:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save({"train_feats": train_feats, "train_labels": train_labels,
                        "test_feats": test_feats, "test_labels": test_labels}, cache_path)
            print(f"[cache] 特征已缓存: {cache_path}")

    # 训练分类头
    print(f"[train] 训练 {args.head} 头 ({args.epochs} epochs) ...")
    best_acc, best_epoch = train_head(train_feats, train_labels,
                                      test_feats, test_labels,
                                      num_classes, args, in_dim=train_feats.shape[1])

    print("=" * 60)
    print(f"结果 | MuM({os.path.basename(args.weights)}) + {args.head} "
          f"[{args.feat}] @ {args.dataset}")
    print(f"Top-1 准确率: {best_acc * 100:.2f}% (best epoch {best_epoch + 1})")
    print("=" * 60)

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
