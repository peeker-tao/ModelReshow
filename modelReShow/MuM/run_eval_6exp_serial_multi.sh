#!/bin/bash
# 3 模型 × (MLP 探针 + 微调) 在 CIFAR10 上评测
# 任务串行, 但每个任务用 6 张卡 (DataParallel 数据并行)

cd /data1/Taohy/ModelReshow/modelReShow/MuM
PY=/data2/student1_ly/miniconda3/envs/thy_mum/bin/python
B=/data1/Taohy/ModelReshow/modelReShow/MuM
GPUS=0,1,2,3,4,5
DEVICE=cuda:0

echo "=============================================================="
echo "6 实验串行 @ CIFAR10 | 每任务 6 卡 (DataParallel: $GPUS) | 开始 $(date)"
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
      --device $DEVICE --gpus $GPUS --feat-batch 96
  echo ">>> PROBE   [$name] 完成: $(date)"

  echo ""
  echo "############ FINETUNE [$name]  $w  ############"
  $PY eval_classification.py --dataset cifar10 --head mlp --weights "$w" \
      --finetune --device $DEVICE --gpus $GPUS --ft-batch-size 96 --ft-epochs 5
  echo ">>> FINETUNE [$name] 完成: $(date)"
done

echo ""
echo "=============================================================="
echo "全部完成: $(date)"
echo "=============================================================="
