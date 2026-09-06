"""Pre-train a MAE ViT-Base on FLIR thermal (IR) images.

Purpose
-------
DeFusion++'s MCUD stage uses two frozen MAE teachers:
    * teacher2 (VI)  -> ImageNet-pretrained MAE ViT-Base (Facebook official)
    * teacher1 (IR)  -> needs an infrared-domain MAE

No public FLIR-pretrained MAE exists, so we self-supervised pre-train one here
on FLIR thermal images, then hand the resulting checkpoint to ``teacher1_pth``.

Checkpoint format matches what ``MMUCMIModel._load_teacher`` expects:
    ``{'model': model.state_dict(), ...}``  (it grabs ``checkpoint['model']``
    and loads with ``strict=False``, so decoder keys are optional).

Normalization matches the MCUD forward path
    ``normalize(img, [0.485,0.456,0.406], [0.229,0.224,0.225])``
so the teacher sees the same input distribution at pre-train and at MCUD.

Usage
-----
    python mae_pretrain_flir.py --data_dir <flir_thermal_root> \
        --output pretrained/mae_flir_vit_base.pth \
        --epochs 400 --batch_size 256 --lr 1.5e-4 \
        --device cuda:0 --log_interval 20 --save_interval 100
"""
import os
import argparse
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image

from models.models_mae import mae_vit_base_patch16


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ThermalDataset(Dataset):
    """Recursively collect all images under a directory as thermal frames.

    Thermal JPEGs are stored as 3-channel (identical R/G/B), so we load RGB and
    keep the standard 3-channel MAE pipeline.
    """

    EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    def __init__(self, root, transform=None, max_samples=None):
        self.paths = []
        for dirpath, _, filenames in os.walk(root, followlinks=True):
            for fn in filenames:
                if fn.lower().endswith(self.EXTS):
                    self.paths.append(os.path.join(dirpath, fn))
        self.paths = sorted(self.paths)
        if max_samples and max_samples > 0:
            self.paths = self.paths[:max_samples]
        self.transform = transform
        if len(self.paths) == 0:
            raise RuntimeError(f'No images found under {root}')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img


def build_transform(train=True):
    if train:
        return T.Compose([
            T.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, base_lr):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='pretrained/mae_flir_vit_base.pth')
    parser.add_argument('--init', type=str, default='',
                        help='path to ImageNet MAE checkpoint for encoder init (domain adaptation)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--warmup_epochs', type=int, default=40)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--max_samples', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--save_interval', type=int, default=100)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    ds = ThermalDataset(args.data_dir, transform=build_transform(True),
                        max_samples=args.max_samples)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        drop_last=True)
    print(f'[MAE-FLIR] images: {len(ds)}, steps/epoch: {len(loader)}')

    model = mae_vit_base_patch16()
    model.to(device)

    # Optional: initialize encoder (and any present decoder keys) from ImageNet MAE.
    if args.init and os.path.exists(args.init):
        ckpt = torch.load(args.init, map_location='cpu')
        state = ckpt['model'] if 'model' in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        n_loaded = len(state) - len(unexpected)
        print(f'[MAE-FLIR] initialized {n_loaded}/{len(state)} keys from {args.init}')
        print(f'[MAE-FLIR] missing (random init): {len(missing)} keys')

    # Stage-1 MAE self-supervised training uses the *full* model (encoder+decoder)
    # to reconstruct masked patches. Only the encoder is later used as teacher1.
    model.train()
    param_groups = [{'params': p, 'weight_decay': args.weight_decay}
                    for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    total_steps = len(loader) * args.epochs
    warmup_steps = len(loader) * args.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, args.lr)

    global_step = 0
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for step, imgs in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)

            # ---- MAE forward: mask -> encode -> decode -> reconstruct ----
            # NOTE: forward_encoder in models_mae.py has masking commented out, so
            # we reproduce the full MAE pipeline here manually.
            x = model.patch_embed(imgs)
            x = x + model.pos_embed[:, 1:, :]
            x, mask, ids_restore = model.random_masking(x, args.mask_ratio)
            cls_token = model.cls_token + model.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            for blk in model.blocks:
                x = blk(x)
            x = model.norm(x)

            pred = model.forward_decoder(x, ids_restore)  # [N, L, p*p*3]
            loss = model.forward_loss(imgs, pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_loss += loss.item()

            if global_step % args.log_interval == 0:
                print(f'[MAE-FLIR] epoch {epoch} step {step} '
                      f'loss {loss.item():.4f} lr {scheduler.get_last_lr()[0]:.2e}')

        avg = epoch_loss / max(1, len(loader))
        print(f'[MAE-FLIR] === epoch {epoch} done, avg loss {avg:.4f} ===')

        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            ckpt = {'model': model.state_dict(), 'epoch': epoch + 1, 'args': vars(args)}
            torch.save(ckpt, args.output)
            print(f'[MAE-FLIR] saved checkpoint to {args.output}')


if __name__ == '__main__':
    main()
