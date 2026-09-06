#!/bin/bash
# 3 模型 × (MLP 探针 + 微调) 在 CIFAR10 上的对比评测
# 串行执行, 每个实验独立输出, 结果追加到日志

cd /data1/Taohy/ModelReshow/modelReShow/MuM
PY=/data2/student1_ly/miniconda3/envs/thy_mum/bin/python
DEVICE=cuda:0
B=/data1/Taohy/ModelReshow/modelReShow/MuM

echo "=============================================================="
echo "开始: 3 模型 × (probe + finetune) @ CIFAR10  |  $(date)"
echo "=============================================================="

for spec in \
  "official:pretrained" \
  "vit_base:$B/checkpoints_traveluav/checkpoint-last.pth" \
  "vit_large:$B/checkpoints_traveluav_256/checkpoint-last.pth"; do

  name="${spec%%:*}"
  w="${spec#*:}"

  echo ""
  echo "############ PROBE   [$name]  $w  ############"
  $PY eval_classification.py --dataset cifar10 --head mlp --weights "$w" --device $DEVICE
  echo ">>> PROBE   [$name] 完成: $(date)"

  echo ""
  echo "############ FINETUNE [$name]  $w  ############"
  $PY eval_classification.py --dataset cifar10 --head mlp --weights "$w" --finetune --device $DEVICE
  echo ">>> FINETUNE [$name] 完成: $(date)"
done

echo ""
echo "=============================================================="
echo "全部完成: $(date)"
echo "=============================================================="
