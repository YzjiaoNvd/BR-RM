# Hallucination Detection Training with GRPO
import argparse
import os
import json
import torch
import pprint
import numpy as np
from tqdm import tqdm
from typing import Any, Optional, Dict, List
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf
from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.hf_datasets.fact_benchmark import HallucinationDataset
from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

# Import our hallucination detection components
from nemo_rl.environments.fact_environment import (
    HallucinationEnvironment,
    HallucinationMetadata,
    format_claim_extraction_prompt,
)

# ========================= DATA PROCESSOR =========================

def hallucination_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process data for hallucination detection training."""
    
    # Extract data
    context = datum_dict.get("context", "")
    response = datum_dict.get("response", "")
    ground_truth_errors = datum_dict.get("ground_truth_errors", "")
    ground_truth_errors = ground_truth_errors.split(" ### ")
    
    # For GRPO, we always start with the claim extraction stage
    claim_prompt = format_claim_extraction_prompt(context, response)
    
    # Create message log for claim extraction stage
    message_log = []
    user_message = {
        "role": "user",
        "content": claim_prompt,
    }
    
    # Apply chat template
    message: list[str] = tokenizer.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    
    user_message["token_ids"] = tokenizer(message, return_tensors="pt")["input_ids"][0]
    user_message["content"] = message
    
    message_log.append(user_message)
    
    # Calculate total length
    total_length = sum(len(msg["token_ids"]) for msg in message_log)
    
    # Check if we need to truncate
    loss_multiplier = 1.0
    if total_length > max_seq_length:
        for msg in message_log:
            msg["token_ids"] = msg["token_ids"][:min(4, max_seq_length // len(message_log))]
        loss_multiplier = 0.0
    
    # Prepare metadata for hallucination environment
    metadata: HallucinationMetadata = {
        "context": context,
        "response": response,
        "ground_truth_errors": ground_truth_errors,
        "claim_extraction_complete": False,
        "extracted_claims": None,
        "search_results": None,
        "num_tool_calls": 0,
    }
    
    return DatumSpec(
        message_log=message_log,
        length=total_length,
        extra_env_info=metadata,
        loss_multiplier=loss_multiplier,
        idx=idx,
        task_name="hallucination_detection",
    )

# ========================= TRAINING SETUP =========================

def setup_hallucination_data(tokenizer, data_config, env_configs):
    """Setup data for hallucination detection training."""
    
    print("\n▶ Setting up hallucination detection data...")
    
    # Create task spec
    hallucination_task_spec = TaskDataSpec(
        task_name="hallucination_detection",
        prompt_file=data_config.get("prompt_file"),
        system_prompt_file=data_config.get("system_prompt_file"),
    )
    
    # Load datasets
    train_data_path = data_config.get("train_data_path")
    val_data_path = data_config.get("val_data_path")
    
    train_dataset = HallucinationDataset(
        train_data_path, 
        task_name="hallucination_detection", 
        shuffle_seed=data_config.get("shuffle_seed_for_training"),
        split='train'
    )
    
    val_dataset = None
    if val_data_path:
        val_dataset = HallucinationDataset(
            val_data_path,
            task_name="hallucination_detection",
            split='validation'
        )
    
    # Setup task data processors
    task_data_processors = defaultdict(
        lambda: (hallucination_task_spec, hallucination_data_processor)
    )
    task_data_processors["hallucination_detection"] = (hallucination_task_spec, hallucination_data_processor)
    

    # Setup hallucination environment
    from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
    
    hallucination_env = HallucinationEnvironment.options(
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.fact_environment.HallucinationEnvironment"
            ),
            "env_vars": dict(os.environ),
        }
    ).remote(env_configs.get("hallucination", {}))
    
    # Update task to environment mapping
    task_to_env = defaultdict(lambda: hallucination_env)
    task_to_env["hallucination_detection"] = hallucination_env
    
    val_task_to_env = defaultdict(lambda: hallucination_env)
    val_task_to_env["hallucination_detection"] = hallucination_env
    
    # Create processed datasets
    processed_train_dataset = AllTaskProcessedDataset(
        train_dataset,
        tokenizer,
        hallucination_task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )
    
    processed_val_dataset = None
    if val_dataset:
        processed_val_dataset = AllTaskProcessedDataset(
            val_dataset,
            tokenizer,
            hallucination_task_spec,
            task_data_processors,
            max_seq_length=data_config["max_input_seq_length"],
        )
    
    return processed_train_dataset, processed_val_dataset, task_to_env, val_task_to_env

# ========================= MAIN TRAINING SCRIPT =========================

def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run GRPO training for Hallucination Detection"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides

def main():
    """Main entry point."""
    args, overrides = parse_args()
    
    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_fact.yaml"
        )
    
    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    
    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)
    
    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")
    
    # Print config
    print("Final config:")
    pprint.pprint(config)
    
    # Get the next experiment directory
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    
    # Initialize Ray
    from nemo_rl.distributed.virtual_cluster import init_ray
    init_ray()
    
    # Setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )
    
    # Setup data
    train_dataset, val_dataset, task_to_env, val_task_to_env = setup_hallucination_data(
        tokenizer,
        config["data"],
        config["env"],
    )
    
    # Setup training using the existing setup function
    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, train_dataset, val_dataset)
    
    # Run GRPO training
    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        val_task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )

if __name__ == "__main__":
    main()