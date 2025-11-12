#!/bin/bash
set -e

# Arguments
STEP_DIR="$1"
STEP_NUM="$2"
RESULTS_DIR="${3:-$(dirname "$STEP_DIR")/outputs}"
DATASETS="${4:-"rewardbench,rmbench,rmb"}"

# Config
GPFS="/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_sft/users/yizhuj/BR-RM"
# CONTAINER="/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_sft/users/yizhuj/nemorl/containers/anyscale+ray+2.43.0-py312-cu125_uv.sqsh"
CONTAINER="/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_sft/users/yizhuj/NeMo-RL/container/nemo-rl:main-3e5481f.squashfs"

# SLURM Config
ACCOUNT="llmservice_modelalignment_sft"
PARTITION="batch_block1"
TIME="4:00:00"
NODES=2
GPUS=8

# Paths
mkdir -p "$RESULTS_DIR/logs"
LOG="$RESULTS_DIR/logs/step_${STEP_NUM}.out"
ERR="$RESULTS_DIR/logs/step_${STEP_NUM}.err"

# Check if already done for all datasets
ALL_DONE=true
IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"
for dataset in "${DATASET_ARRAY[@]}"; do
    dataset=$(echo "$dataset" | xargs)  # trim whitespace
    OUTPUT="$RESULTS_DIR/step_${STEP_NUM}_${dataset}_results.json"
    
    if [ ! -f "$OUTPUT" ] || ! jq empty "$OUTPUT" >/dev/null 2>&1; then
        ALL_DONE=false
        break
    fi
done

if [ "$ALL_DONE" = true ]; then
    echo "✓ Step $STEP_NUM already evaluated for all datasets: $DATASETS"
    exit 0
fi

# Create command to run all datasets sequentially
COMMAND="cd $GPFS && "
for dataset in "${DATASET_ARRAY[@]}"; do
    dataset=$(echo "$dataset" | xargs)  # trim whitespace
    OUTPUT="$RESULTS_DIR/step_${STEP_NUM}_${dataset}_results.json"
    COMMAND+="echo 'Processing dataset: $dataset' && "
    COMMAND+="uv run python examples/run_eval_brrm.py --dataset=${dataset} ++generation.model_name=${STEP_DIR} ++eval.output_file=${OUTPUT} ++eval.batch_size=1024 ++generation.vllm_cfg.tensor_parallel_size=1 ++generation.vllm_cfg.gpu_memory_utilization=0.95 ++cluster.gpus_per_node=${GPUS} ++cluster.num_nodes=${NODES} && "
done
# Remove the trailing &&
COMMAND=${COMMAND%"&& "}

# Set up environment and submit job using the new format
cd "$GPFS"

HF_HOME=/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_sft/users/yizhuj/hf_cache \
COMMAND="$COMMAND" \
CONTAINER="$CONTAINER" \
MOUNTS="$GPFS:$GPFS,/lustre:/lustre" \
sbatch \
    --nodes=$NODES \
    --account=$ACCOUNT \
    --job-name=eval_brrm_step${STEP_NUM}_multi \
    --partition=$PARTITION \
    --time=$TIME \
    --gres=gpu:$GPUS \
    --output=$LOG \
    --error=$ERR \
    ray.sub

echo "📤 Submitted job for step_$STEP_NUM with datasets: $DATASETS"