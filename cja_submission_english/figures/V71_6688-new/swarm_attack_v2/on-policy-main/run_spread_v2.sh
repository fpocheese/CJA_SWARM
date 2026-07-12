#!/bin/bash
# Simple Spread V2 多算法对比训练
cd /home/uav/00gao_xueshu/togsy_2025/0620septimedone/on-policy-main
SCRIPT="onpolicy/scripts/train_spread_v2.py"
EP=1000

echo "============================================"
echo " Simple Spread V2 - 6算法 × 3种子 × ${EP} ep"
echo "============================================"

for ALGO in MAPPO Advanced-MAPPO IPPO IA2C IQL MADDPG; do
    echo ""
    echo ">>> ${ALGO} (3 seeds 并行)"
    for SEED in 1 2 3; do
        python -u ${SCRIPT} --algo ${ALGO} --seed ${SEED} --num_episodes ${EP} \
            > /tmp/v2_${ALGO}_s${SEED}.log 2>&1 &
    done
    wait
    echo ">>> ${ALGO} done!"
    for SEED in 1 2 3; do tail -1 /tmp/v2_${ALGO}_s${SEED}.log; done
done

echo ""
echo ">>> Generating plots..."
python -u ${SCRIPT} --plot_only
echo "All done!"
