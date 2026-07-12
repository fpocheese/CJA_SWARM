#!/bin/bash
# Simple Spread 多算法对比训练脚本
# 6算法 × 3种子 = 18次训练
# 并行运行（每次3个进程）

cd /home/uav/00gao_xueshu/togsy_2025/0620septimedone/on-policy-main

SCRIPT="onpolicy/scripts/train_simple_spread.py"
EPISODES=1000

echo "============================================"
echo " Simple Spread 多算法对比训练"
echo " 6算法 × 3种子 × ${EPISODES} episodes"
echo "============================================"

# 逐个算法运行（每个算法的3个seed并行）
for ALGO in MAPPO Advanced-MAPPO IPPO IA2C IQL MADDPG; do
    echo ""
    echo ">>> 开始训练: ${ALGO} (3 seeds 并行)"
    
    for SEED in 1 2 3; do
        python -u ${SCRIPT} --algo ${ALGO} --seed ${SEED} --num_episodes ${EPISODES} \
            > /tmp/train_${ALGO}_seed${SEED}.log 2>&1 &
    done
    
    # 等待当前算法的3个seed完成
    wait
    
    echo ">>> ${ALGO} 全部完成!"
    # 打印最后几行
    for SEED in 1 2 3; do
        tail -3 /tmp/train_${ALGO}_seed${SEED}.log
    done
done

echo ""
echo "============================================"
echo " 全部训练完成! 开始生成对比图..."
echo "============================================"

# 生成对比图
python ${SCRIPT} --plot_only

echo "完成!"
