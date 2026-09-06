#!/bin/bash
# 3 模型 × (MLP 探针 + 微调) 在 CIFAR10 上并行评测, 每实验独占一张卡
# GPU 0-5 空闲; 每个实验独立日志, 最后统一汇总结果

cd /data1/Taohy/ModelReshow/modelReShow/MuM
PY=/data2/student1_ly/miniconda3/envs/thy_mum/bin/python
B=/data1/Taohy/ModelReshow/modelReShow/MuM
LOG=$B/eval_logs
mkdir -p "$LOG"

run() {
  local name=$1 type=$2 w=$3 dev=$4
  local flag=""
  [ "$type" = "finetune" ] && flag="--finetune"
  $PY eval_classification.py --dataset cifar10 --head mlp --weights "$w" $flag --device "$dev" \
    > "$LOG/${name}_${type}.log" 2>&1
  local acc
  acc=$(grep -oP 'Top-1 准确率: \K[0-9.]+' "$LOG/${name}_${type}.log" | tail -1)
  printf '[%-10s / %-8s] 完成 %s | Top-1 = %s%%\n' "$name" "$type" "$(date '+%H:%M:%S')" "$acc"
}

echo "=============================================================="
echo "6 实验并行 @ CIFAR10 | 开始 $(date)"
echo "  official   probe/finetune -> cuda:0 / cuda:1"
echo "  vit_base   probe/finetune -> cuda:2 / cuda:3"
echo "  vit_large  probe/finetune -> cuda:4 / cuda:5"
echo "=============================================================="

run official  probe     pretrained                                            cuda:0 &
run official  finetune  pretrained                                            cuda:1 &
run vit_base  probe     "$B/checkpoints_traveluav/checkpoint-last.pth"        cuda:2 &
run vit_base  finetune  "$B/checkpoints_traveluav/checkpoint-last.pth"        cuda:3 &
run vit_large probe     "$B/checkpoints_traveluav_256/checkpoint-last.pth"    cuda:4 &
run vit_large finetune  "$B/checkpoints_traveluav_256/checkpoint-last.pth"    cuda:5 &

wait

echo "=============================================================="
echo "全部完成 $(date) —— 结果汇总:"
echo "=============================================================="
for f in "$LOG"/*.log; do
  acc=$(grep -oP 'Top-1 准确率: \K[0-9.]+' "$f" | tail -1)
  printf '  %-30s Top-1 = %s%%\n' "$(basename "$f" .log)" "$acc"
done
