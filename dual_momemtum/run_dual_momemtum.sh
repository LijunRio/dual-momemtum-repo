#!/bin/bash

# Usage: bash run_dual_momemtum.sh [-n iterations] [-k api-key]

# ============================================================================
# Configuration (overrides dual_momemtum_config.py)
# ============================================================================

ITERATIONS=20

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"
API_KEY="${OPENROUTER_API_KEY:-}"

TRAIN_DATA_PATH="${DATA_DIR}/dev_nova_100.json"
DEV_DATA_PATH="${DATA_DIR}/dev_nova_100.json"
TEST_DATA_PATH="${DATA_DIR}/dev_nova_100.json"

IMAGE_PATH_FROM="/home/june/datasets/nova"
IMAGE_PATH_TO="/home/june/datasets/nova"

# # BTD dataset (uncomment to use)
# TRAIN_DATA_PATH="${DATA_DIR}/dev_btd_100.json"
# DEV_DATA_PATH="${DATA_DIR}/dev_btd_100.json"
# TEST_DATA_PATH="${DATA_DIR}/dev_btd_100.json"
# IMAGE_PATH_FROM="/home/june/datasets/brain_tumor_dataset"
# IMAGE_PATH_TO="/home/june/datasets/brain_tumor_dataset"



BASE_OUTPUT_DIR="./output"
declare -a MODEL_PATHS=(
    # "/home/june/cache/huggingface_checkpoints/Qwen2.5-VL-3B-Instruct"
    "/home/june/cache/huggingface_checkpoints/Qwen2.5-VL-7B-Instruct"
    # "/home/june/cache/huggingface_checkpoints/Qwen2.5-VL-32B-Instruct"
    # "/home/june/cache/huggingface_checkpoints/Qwen2.5-VL-72B-Instruct"
)

# Ablation mode: language_only | visual_only | (empty = dual)
ABLATION_MODE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--iterations) ITERATIONS="$2"; shift 2 ;;
        -k|--api-key) API_KEY="$2"; shift 2 ;;
        --ablation) ABLATION_MODE="$2"; shift 2 ;;
        --train-data) TRAIN_DATA_PATH="$2"; shift 2 ;;
        --dev-data) DEV_DATA_PATH="$2"; shift 2 ;;
        --test-data) TEST_DATA_PATH="$2"; shift 2 ;;
        --image-path-from) IMAGE_PATH_FROM="$2"; shift 2 ;;
        --image-path-to) IMAGE_PATH_TO="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash run_dual_momemtum.sh [options]"
            echo ""
            echo "Options:"
            echo "  -n, --iterations NUM     Number of iterations"
            echo "  -k, --api-key KEY        API key"
            echo "  --ablation MODE          Ablation mode: language_only, visual_only"
            echo "  --train-data PATH        Training data path"
            echo "  --dev-data PATH          Dev set data path"
            echo "  --test-data PATH         Test set data path"
            echo "  --image-path-from PATH   Image path replacement source"
            echo "  --image-path-to PATH     Image path replacement target"
            echo "  -h, --help               Show this help"
            echo ""
            echo "Defaults:"
            echo "  iterations:       $ITERATIONS"
            echo "  train-data:       $TRAIN_DATA_PATH"
            echo "  dev-data:         $DEV_DATA_PATH"
            echo "  test-data:        $TEST_DATA_PATH"
            echo "  image-path-from:  $IMAGE_PATH_FROM"
            echo "  image-path-to:    $IMAGE_PATH_TO"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

normalize_model_name() {
    basename "$1" | sed 's/-Instruct$//' | tr '[:upper:]' '[:lower:]'
}

echo "Dual Momemtum - iterations: $ITERATIONS | models: ${#MODEL_PATHS[@]}"
echo ""

SUCCESS=0
FAILED=0

for model_path in "${MODEL_PATHS[@]}"; do
    model_name=$(basename "$model_path")
    normalized_name=$(normalize_model_name "$model_path")
    output_dir="${BASE_OUTPUT_DIR}/${normalized_name}"

    echo "Running $model_name ..."

    cmd="python run_dual_momemtum.py -n $ITERATIONS --output-dir $output_dir --model-path $model_path"
    [ -n "$API_KEY" ] && cmd="$cmd --api-key $API_KEY"
    [ -n "$ABLATION_MODE" ] && cmd="$cmd --ablation $ABLATION_MODE"
    [ -n "$TRAIN_DATA_PATH" ] && cmd="$cmd --train-data $TRAIN_DATA_PATH"
    [ -n "$DEV_DATA_PATH" ] && cmd="$cmd --dev-data $DEV_DATA_PATH"
    [ -n "$TEST_DATA_PATH" ] && cmd="$cmd --test-data $TEST_DATA_PATH"
    [ -n "$IMAGE_PATH_FROM" ] && cmd="$cmd --image-path-from $IMAGE_PATH_FROM"
    [ -n "$IMAGE_PATH_TO" ] && cmd="$cmd --image-path-to $IMAGE_PATH_TO"

    if eval "$cmd"; then
        echo "Done: $model_name"
        ((SUCCESS++))
    else
        echo "Failed: $model_name"
        ((FAILED++))
    fi

    echo ""
done

echo "========================================"
echo "Results: success=$SUCCESS failed=$FAILED"
echo "Output: $BASE_OUTPUT_DIR"
echo "========================================"

[ $FAILED -eq 0 ] && exit 0 || exit 1
