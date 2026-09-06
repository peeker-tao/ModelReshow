# DeFusion++ — Complete Reproduction

本仓库是对论文 **《Fusion from Decomposition: A Self-Supervised Approach for Image Fusion and Beyond》**（DeFusion++）的**完整可运行复现**，基于残缺项目 `DeFusion-plusplus` 与论文内容重建，未改动原残缺项目。

## 1. 方法概述

DeFusion++ 是一个基于 **ViT + Cross-Attention** 的自监督图像融合框架，通过两个 pretext task 学习「公共信息（common）」与「特有信息（unique）」的分解：

- **CUD（Common-Unique Decomposition）**：单模态自监督。对同一张图施加 grid mask 得到两个视角，分解出 common / upper / lower 三个分量，并重构融合图。
- **MCUD（Multi-modal Common-Unique Decomposition）**：多模态自监督。用两个冻结的 MAE 教师（IR / VI）引导，将不同模态的 latent 对齐到公共 / 特有子空间。

### 网络结构

| 组件 | 说明 |
|------|------|
| encoder | `tinymim_vit_tiny_patch16`（embed_dim=192, depth=12, num_heads=6, patch=16） |
| 教师模型 | `mae_vit_base_patch16`（embed_dim=768, depth=12, 返回 cls token 序列），MCUD 阶段冻结 |
| 分解 decoder | cross-attention `TransformerAttenBlockVpaper`（decoder_embed_dim=192, decoder_img_dim=768, num_heads=16） |
| 融合 decoder | `recon_blocks_mim_encoder` + fuse blocks |
| 归一化 | mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]` |

## 2. 目录结构

```
DeFusion-plusplus-complete/
├── models/                # 模型定义
│   ├── vit.py             # PatchEmbed / Block
│   ├── tinymim.py         # TinyMIM encoder（含动态 pos_embed 插值）
│   ├── models_mae.py      # MAE 教师模型
│   ├── multiAtten.py      # cross-attention block
│   ├── pos_embed.py       # 2D sincos position embedding
│   ├── UCMIModel.py       # Stage-1 单模态 CUD 训练模型
│   ├── MMUCMIModel.py     # Stage-2 多模态 MCUD 训练模型
│   └── MUCMIModelTest.py  # 推理模型（无教师/无 mask 分支）
├── loss/                  # 损失函数
│   ├── mix_fp_loss.py     # SelfTrainLoss（CUD）
│   ├── multimodalloss.py  # MultiModalLoss（MCUD）
│   └── simple_loss.py     # PretrainLoss
├── data/                  # 数据集
│   ├── self_mixpretrain_dataset.py   # 单模态 grid-mask 自监督数据
│   ├── irvis_train_dataset.py        # 成对 IR/VI 训练数据（MCUD）
│   ├── visir_fusion_dataset.py       # IR/VI 测试数据
│   ├── multi_exposure_dataset.py     # 多曝光测试数据
│   └── multi_focus_dataset.py        # 多聚焦测试数据
├── option/                # 训练/测试配置（yaml）
├── utils/                 # util / build_code_arch / pos_embed
├── pretrained/            # 预训练权重（含 demo_random.pth）
├── selftrain.py           # Stage-1 CUD 训练入口
├── selftrain_multimodal.py# Stage-2 MCUD 训练入口
├── test.py                # 推理入口
├── generate_demo_weight.py# 生成随机 demo 权重
└── requirement.txt
```

## 3. 环境安装

已用 conda 环境 `Defusion-plusplus`（Python 3.9）验证：

```bash
conda create -n Defusion-plusplus python=3.9
conda activate Defusion-plusplus
pip install -r requirement.txt
```

> 注意：`torch/torchvision` 版本为 `2.1.2+cu118 / 0.16.2+cu118`，如需其他 CUDA 版本请到 [pytorch.org](https://pytorch.org) 安装对应 wheel。

## 4. 推理

```bash
python test.py -opt option/test/MSRS_demo.yaml
```

对 MSRS 测试集（`test/ir` + `test/vi`）逐对输出 6 张图：`common / upper / lower / over / under / recover`，保存到 `results/<name>/test_images/`。

### 关于权重

- 仓库附带的 `pretrained/demo_random.pth` 为**随机初始化权重**（`MUCMIMNetTest`），仅用于验证推理流程可跑通，**无融合效果**。
- 若要获得论文效果，需按第 5 节训练，或将训练得到的 checkpoint 转为推理格式（见下）。

## 5. 训练

### Stage 1：单模态 CUD（自监督预训练）

```bash
python selftrain.py -opt option/train/MSRS_CUD_demo.yaml
```

论文原始配置使用 **COCO（约 11.8 万张图）** 做 CUD 预训练，见 `option/train/SelfTrained_SDatasetFast.yaml`。仓库额外提供 `MSRS_CUD_demo.yaml`（使用 MSRS train/ir，1083 张）用于快速验证训练流程。

### Stage 2：多模态 MCUD

```bash
python selftrain_multimodal.py -opt option/train/MSRS_MCUD_demo.yaml
```

论文原始配置见 `option/train/SelfTrained_MDatasetFast.yaml`（CUD 分支用 COCO，MCUD 分支用 MSRS train 对）。

### 训练结果 → 推理权重转换

推理模型 `MUCMIMNetTest` 是训练模型 `MUCMIMNet` 的**推理子集**（不含 mask_token / mim decoder / 教师）。训练 checkpoint 可直接被推理模型以 `strict=False` 加载，或在训练脚本中保存过滤后的权重。最简单方式：训练结束后，用以下脚本提取推理权重：

```python
import torch
from models.MMUCMIModel import MUCMIMNet
from models.MUCMIModelTest import MUCMIMNetTest

ckpt = torch.load('experiments/<name>/models/latest.pth', map_location='cpu')
test_model = MUCMIMNetTest()
# 过滤掉训练专用组件（mask_token / mim_decoder / decoder_pos_embed / 教师）
state = {k: v for k, v in ckpt['state_dict'].items()
         if not any(s in k for s in ('mask_token', 'decoder_pos_embed', 'mim_decoder', 'model_encoder'))}
test_model.load_state_dict(state, strict=False)
torch.save({'state_dict': test_model.state_dict(), 'epoch': ckpt.get('epoch')}, 'pretrained/DeFusionpp.pth')
```

然后将 `option/test/MSRS_demo.yaml` 中的 `resume_state` 指向该权重即可。

## 6. 关键修复说明（相对残缺项目）

1. **`fuse_main` 未定义**：原 `MMUCMIModel.py` 存在 `fuse_img = fuse_img + fuse_main` 而 `fuse_main` 未定义的 bug，已在 `UCMIModel.py` / `MMUCMIModel.py` / `MUCMIModelTest.py` 中统一修复。
2. **输出范围不匹配**：原模型输出归一化空间，而损失目标在 `[0,1]`，三个模型均增加反归一化 `_denormalize`。
3. **PatchEmbed 尺寸断言**：移除 `vit.py` 中的 `assert H == img_size[0]`，支持任意输入尺寸。
4. **pos_embed 固定 224**：`tinymim.py` 增加 `interpolate_pos_embed`，动态插值到实际 patch 网格。
5. **路径过滤**：`utils/util.py` 的 `mkdirs` 与 `test.py` 增加 `isinstance(path, str)` 及 `pretrain`/`resume` 键过滤，避免 `strict_load`（bool）与 `/home/xxx` 占位路径触发报错。

## 7. 参考

- 论文：Liang et al., *Fusion from Decomposition: A Self-Supervised Approach for Image Fusion and Beyond* (2024).
- 原残缺项目：`../DeFusion-plusplus/`（本复现未改动）。
