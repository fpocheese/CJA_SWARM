#!/usr/bin/env bash
# 批量搜索 3 个突防成功的 seed, 输出到 outputs/multi_success/seed_<N>/
# 用法:  bash scripts/find_3_successes.sh [start_seed] [end_seed]
set -u
START=${1:-0}
END=${2:-100}
TARGET=3
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE="$ROOT/outputs/multi_success"
mkdir -p "$BASE"

CONDA="$HOME/miniconda3/bin/conda"
PY="$CONDA run --no-capture-output -n rlgpu python"

success=0
attempted=0
SUCC_LIST="$BASE/successes.txt"
: > "$SUCC_LIST"

for seed in $(seq "$START" "$END"); do
    if [ "$success" -ge "$TARGET" ]; then
        break
    fi
    attempted=$((attempted+1))
    OUT="$BASE/seed_${seed}"
    echo "============== [$attempted] seed=$seed (success so far: $success/$TARGET) =============="
    rm -rf "$OUT"
    set +e
    LOG="$OUT.log"
    mkdir -p "$OUT"
    $PY "$ROOT/scripts/analyze_ep02_allocation.py" \
        --seed "$seed" --out-dir "$OUT" > "$LOG" 2>&1
    rc=$?
    set -e
    tail -3 "$LOG"
    EVT="$OUT/ep_events.json"
    if [ -f "$EVT" ] && grep -q '"result": "SUCCESS"' "$EVT"; then
        success=$((success+1))
        echo "$seed" >> "$SUCC_LIST"
        echo ">>> SUCCESS #$success at seed=$seed (saved to $OUT)"
    elif [ -f "$EVT" ] && grep -q '"result": "MISS"' "$EVT"; then
        echo "    miss, removed $OUT"
        rm -rf "$OUT" "$LOG"
    else
        echo "    [warn] script error rc=$rc at seed=$seed (log kept at $LOG)"
        rm -rf "$OUT"
    fi
done

echo
echo "================================================================"
echo "Done. Attempted=$attempted, Successes=$success/$TARGET"
echo "Successful seeds:"
cat "$SUCC_LIST"
ls -lh "$BASE"
