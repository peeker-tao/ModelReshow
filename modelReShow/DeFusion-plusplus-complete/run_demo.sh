#!/bin/bash
# DeFusion++ MCUD demo 训练启动脚本
# - 自动选择利用率最低（最空闲）的 GPU，也可用 GPU_ID 环境变量手动指定
# - 用 setsid + nohup 后台运行，SSH 断开也不中断
set -e
cd /data1/Taohy/ModelReshow/modelReShow/DeFusion-plusplus-complete

source /data2/student1_ly/miniconda3/etc/profile.d/conda.sh
conda activate Defusion-plusplus

if [ -z "$GPU_ID" ]; then
  # 自动检测：利用率最低的 GPU（最空闲）
  GPU_ID=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | \
           sort -t',' -k2 -n | head -1 | cut -d',' -f1 | tr -d ' ')
fi
echo "使用 GPU: ${GPU_ID}"
export CUDA_VISIBLE_DEVICES=${GPU_ID}

LOG_FILE="run_demo_$(date +%Y%m%d_%H%M%S).log"

# setsid 脱离会话 + nohup 忽略挂断信号，后台运行
setsid nohup python selftrain_multimodal.py -opt option/train/COCO_MSRS_MCUD_demo.yaml \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "训练已后台启动，PID: ${PID}"
echo "日志文件: ${LOG_FILE}"
echo "查看日志: tail -f $(pwd)/${LOG_FILE}"
