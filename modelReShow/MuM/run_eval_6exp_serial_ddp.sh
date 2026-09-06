#!/bin/bash
# 3 模型 × (MLP 探针 + 微调) 在 CIFAR10 上评测
# 任务串行: probe 单卡(快), finetune 用 torchrun + DDP 6 卡 (NCCL allreduce)

cd /data1/Taohy/ModelReshow/modelReShow/MuM
PY=/data2/student1_ly/miniconda3/envs/thy_mum/bin/python
B=/data1/Taohy/ModelReshow/modelReShow/MuM
NPROC=6
MASTER_PORT=29517

echo "=============================================================="
echo "6 实验串行 @ CIFAR10 | finetune 用 DDP $NPROC 卡 | 开始 $(date)"
echo "=============================================================="

for spec in \
  "official:pretrained" \
  "vit_base:$B/checkpoints_traveluav/checkpoint-last.pth" \
  "vit_large:$B/checkpoints_traveluav_256/checkpoint-last.pth"; do

  name="${spec%%:*}"
  w="${spec#*:}"

  echo ""
  echo "############ PROBE   [$name]  $w  ############"
  $PY eval_classification.py --dataset cifar10 --head mlp --weights "$w" \
      --device cuda:0 --feat-batch 96
  echo ">>> PROBE   [$name] 完成: $(date)"

  echo ""
  echo "############ FINETUNE [$name]  $w  ############"
  $PY -m torch.distributed.run --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
      eval_classification.py --dataset cifar10 --head mlp --weights "$w" \
      --finetune --ft-batch-size 16 --ft-epochs 5
  echo ">>> FINETUNE [$name] 完成: $(date)"
done

echo ""
echo "=============================================================="
echo "全部完成: $(date)"
echo "=============================================================="
