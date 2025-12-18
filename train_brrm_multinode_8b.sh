#!/bin/bash

set -x

GPFS="./BR-RM"
CONTAINER=${YOUR_CONTAINER_PATH_HERE}

# Number of nodes for the job
NUM_ACTOR_NODES=2

# Model and training configuration

MODEL="Qwen/Qwen3-8B"
MODEL_NAME="qwen3_8b"


FSDP2=True
ACT_CKPT=True
CPU_OFFLOAD=True
SP=False
lr=5e-7
temp=1
grpo_bs=256
prompts_per_step=128
rollouts_per_prompt=8
kl=0.001

NAME="grpo_brrm_${MODEL_NAME}_lr_${lr}_temp_${temp}_kl_${kl}_bs_${grpo_bs}_rollout_${rollouts_per_prompt}_num_prompts_${prompts_per_step}"

RESULTS_DIR="./BR-RM/results/${NAME}"
mkdir -p $RESULTS_DIR

ACTOR_LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p $ACTOR_LOG_DIR
PPO_ERRFILE="${ACTOR_LOG_DIR}/%j_%t.err"
PPO_OUTFILE="${ACTOR_LOG_DIR}/%j_%t.log"


# Construct the command to run
COMMAND="cd ${GPFS} && ulimit -c 0 && uv run examples/run_grpo_brrm.py \
    ++logger.wandb_enabled=False \
    ++checkpointing.checkpoint_dir=${RESULTS_DIR} \
    ++cluster.num_nodes=${NUM_ACTOR_NODES} \
    policy.dtensor_cfg.enabled=${FSDP2} \
    policy.dtensor_cfg.tensor_parallel_size=1 \
    policy.dtensor_cfg.sequence_parallel=${SP} \
    policy.dtensor_cfg.activation_checkpointing=${ACT_CKPT} \
    policy.dtensor_cfg.cpu_offload=${CPU_OFFLOAD} \
    ++policy.dtensor_cfg.context_parallel_size=1 \
    ++cluster.gpus_per_node=8 \
    grpo.num_prompts_per_step=${prompts_per_step} \
    grpo.num_generations_per_prompt=${rollouts_per_prompt} \
    grpo.val_period=10 \
    grpo.max_val_samples=16 \
    grpo.val_batch_size=16 \
    data.train_data_path="${GPFS}/dataset/train_data.jsonl" \
    data.val_data_path="${GPFS}/dataset/val_data.jsonl" \
    loss_fn.reference_policy_kl_penalty=${kl} \
    loss_fn.use_on_policy_kl_approximation=False \
    loss_fn.use_importance_sampling_correction=False \
    checkpointing.keep_top_k=10 \
    checkpointing.save_period=10 \
    loss_fn.ratio_clip_min=0.2 \
    loss_fn.ratio_clip_max=0.28 \
    policy.model_name=${MODEL} \
    policy.make_sequence_length_divisible_by=2 \
    policy.generation.vllm_cfg.tensor_parallel_size=2 \
    policy.train_global_batch_size=${grpo_bs} \
    policy.train_micro_batch_size=1 \
    policy.generation_batch_size=1 \
    policy.logprob_batch_size=1 \
    policy.max_total_sequence_length=16384 \
    policy.optimizer.kwargs.lr=${lr} \
    policy.optimizer.kwargs.weight_decay=0 \
    policy.generation.temperature=${temp} \
    policy.generation.vllm_cfg.gpu_memory_utilization=0.8 "



# Set up mounts
MOUNTS=${YOUR_MOUNTS}

# Submit job using ray.sub
COMMAND="${COMMAND}" \
CONTAINER="${CONTAINER}" \
MOUNTS="${MOUNTS}" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=${YOUR_CLUSTER_ACCOUNT} \
    --job-name=grpo_brrm_${MODEL_NAME}_${lr} \
    --partition=${YOUR_CLUSTER_PARTITION} \
    --gres=gpu:8 \
    --mem=0 \
    --dependency=singleton \
    -o $PPO_OUTFILE \
    -e $PPO_ERRFILE \
    ray.sub