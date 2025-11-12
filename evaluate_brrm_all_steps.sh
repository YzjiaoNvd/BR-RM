#!/bin/bash
set -e

# Usage: ./evaluate_brrm_all_steps.sh [BASE_DIR] [MODE] [DATASETS] 
# BASE_DIR: Path to the results directory
# DATASETS: Comma-separated list of datasets (e.g., "judgebench,rmbench,rewardbench")
# MODE: "all" (default), "latest"/"last" - evaluate all checkpoints or only the last one

# Base directory (from argument or default)
BASE_DIR="${1:-/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_sft/users/yizhuj/BR-RM/results/2grpo_hs3_16K_step240_clip_max_0.28_llama3.1_8B_lr_2e-6_temp_1_kl_0.001_grpo_bs_256_rollout_16_num_prompts_128}"
MODE="${2:-latest}"  # "all" or "latest"
DATASETS="${3:-"rewardbench,rmbench,rmb"}"
HF_DIR="${BASE_DIR}/HF"
RESULTS_DIR="${BASE_DIR}/outputs"


# Find evaluation script
EVAL_SCRIPT="$(dirname "$0")/evaluate_brrm_one_step.sh"
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "Error: evaluate_genrm_one_step.sh not found"
    exit 1
fi

# Validate mode
if [ "$MODE" != "all" ] && [ "$MODE" != "latest" ] && [ "$MODE" != "last" ]; then
    echo "Error: MODE must be 'all', 'latest', or 'last', got: $MODE"
    exit 1
fi

echo "GenRM Evaluation for: $BASE_DIR"
echo "Datasets: $DATASETS"
echo "Mode: $MODE"
echo "Results: $RESULTS_DIR"
echo "----------------------------------------"

# Find step directories
STEP_DIRS=$(find ${HF_DIR} -type d -name "step_*" 2>/dev/null || find ${BASE_DIR} -type d -name "step_*")

if [ -z "$STEP_DIRS" ]; then
    echo "No checkpoint directories found!"
    exit 1
fi

# Sort step directories
SORTED_STEP_DIRS=$(echo "$STEP_DIRS" | while read dir; do
    # Extract step number from the last part of the path
    step_num=$(basename "$dir" | sed 's/step_//')
    echo "$step_num $dir"
done | sort -n | awk '{print $2}')



# Choose directories based on mode
if [ "$MODE" = "latest" ] || [ "$MODE" = "last" ]; then
    # Get only the latest step
    STEP_DIRS_TO_PROCESS=$(echo "$SORTED_STEP_DIRS" | tail -1)
    echo "Processing only the last checkpoint..."
else
    # Process all steps
    STEP_DIRS_TO_PROCESS="$SORTED_STEP_DIRS"
    echo "Processing all checkpoints..."
fi

# Process each step
for step_dir in $STEP_DIRS_TO_PROCESS; do
    step_num=$(basename "$step_dir" | grep -Eo '[0-9]+')
    
    if [ -z "$step_num" ]; then
        echo "⚠️  Invalid: $(basename "$step_dir")"
        continue
    fi
    
    echo "Processing step_$step_num..."
    bash "$EVAL_SCRIPT" "$step_dir" "$step_num" "$RESULTS_DIR" "$DATASETS"

done