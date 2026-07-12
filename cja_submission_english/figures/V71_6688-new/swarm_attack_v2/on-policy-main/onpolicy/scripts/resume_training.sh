#!/bin/bash
# ============================================================
#  恢复训练脚本 - 只跑剩余的 MADDPG seed3/4/5，然后出图
#  用法: bash resume_training.sh
#  或后台运行: nohup bash resume_training.sh > resume.log 2>&1 &
# ============================================================
set -e

cd /home/uav/00gao_xueshu/togsy_2025/0620septimedone/on-policy-main

echo "============================================"
echo "  恢复v6训练: MADDPG seed 3,4,5"
echo "  $(date)"
echo "============================================"

# 跑剩余的 MADDPG seed3/4/5
conda run -n rlgpu --no-capture-output python onpolicy/scripts/train_simple_converge.py \
    --num_episodes 5000 \
    --seeds 3 4 5 \
    --algorithms MADDPG

echo ""
echo "============================================"
echo "  MADDPG训练完成，开始生成IEEE图表..."
echo "  $(date)"
echo "============================================"

# 用所有已有数据生成最终图
conda run -n rlgpu --no-capture-output python onpolicy/scripts/train_simple_converge.py \
    --plot_only \
    --algorithms Advanced-MAPPO MAPPO IPPO IA2C IQL MADDPG

echo ""
echo "============================================"
echo "  全部完成! $(date)"
echo "  图表位置: onpolicy/scripts/results/simple_converge/"
echo "============================================"
