#!/bin/bash
# =============================================================================
#  多算法对比训练 - 主控脚本
#  5种算法 × 3种子 = 15次训练, 完成后自动生成对比图
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")/.."  # on-policy-main root
echo "Working directory: $(pwd)"

STEPS=${1:-150000}
CONDA_ENV="rlgpu"
TRAIN_SCRIPT="onpolicy/scripts/train_single_algo.py"
PLOT_SCRIPT="onpolicy/scripts/plot_comparison.py"
SAVE_DIR="onpolicy/scripts/results/comparison"
LOG_DIR="/tmp/train_logs"
mkdir -p "$LOG_DIR" "$SAVE_DIR"

ALGOS=("MAPPO" "Advanced-MAPPO" "IPPO" "IA2C" "IQL")
SEEDS=(1 2 3)

echo "============================================================"
echo "  多算法对比训练"
echo "  算法: ${ALGOS[*]}"
echo "  种子: ${SEEDS[*]}"
echo "  步数: $STEPS"
echo "============================================================"

# 逐个运行 (GPU显存有限不适合同时跑多个)
TOTAL=$((${#ALGOS[@]} * ${#SEEDS[@]}))
COUNT=0

for ALGO in "${ALGOS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        COUNT=$((COUNT + 1))
        LOG_FILE="$LOG_DIR/${ALGO}_seed${SEED}.log"
        echo ""
        echo "[$COUNT/$TOTAL] 训练 $ALGO seed=$SEED ..."
        echo "  日志: $LOG_FILE"

        conda run --no-capture-output -n $CONDA_ENV python -u $TRAIN_SCRIPT \
            --algo "$ALGO" --seed "$SEED" --compare_steps "$STEPS" \
            --n_training_threads 4 2>&1 | tee "$LOG_FILE"

        echo "[$COUNT/$TOTAL] $ALGO seed=$SEED 完成!"
    done
done

echo ""
echo "============================================================"
echo "  所有训练完成! 生成对比图..."
echo "============================================================"

conda run --no-capture-output -n $CONDA_ENV python -u $PLOT_SCRIPT \
    --save_dir "$SAVE_DIR" \
    --algos MAPPO Advanced-MAPPO IPPO IA2C IQL \
    --smooth 5 --x_scale 1

echo ""
echo "============================================================"
echo "  全部完成!"
echo "  对比图路径: $SAVE_DIR/"
echo "============================================================"
ls -la "$SAVE_DIR"/*.pdf "$SAVE_DIR"/*.png 2>/dev/null || echo "  (图片文件待检查)"
