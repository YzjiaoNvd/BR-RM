# Two-Stage Unified Quality Assessment GenRM Training
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
from nemo_rl.data.hf_datasets.reward_benchmarks import LocalDataset
from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

# Import our unified quality assessment components
from nemo_rl.environments.brrm_environment import (
    BRRMEnvironment, 
    TwoStageMetadata,
    format_unified_analysis_prompt,  # Updated function name
    format_scoring_stage_prompt,
    parse_scoring_response,
    parse_unified_analysis_response,  # Updated function name
)

# ========================= DATA PROCESSOR =========================

def two_stage_genrm_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process HelpSteer3 data for two-stage unified quality assessment training."""
    
    # Extract data
    context = datum_dict.get("context", "")
    response1 = datum_dict.get("response1", "")
    response2 = datum_dict.get("response2", "")
    
    num_responses = datum_dict.get("num_responses", 2)
    preference = datum_dict.get("preference")

    # For GRPO, we always start with the quality assessment stage
    # The environment will handle the transition to scoring stage
    quality_prompt = format_unified_analysis_prompt(context, response1, response2)  # Updated function call
    
    # Create message log for quality assessment stage
    message_log = []
    user_message = {
        "role": "user",
        "content": quality_prompt,
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
    
    # Prepare metadata for two-stage environment
    metadata: TwoStageMetadata = {
        "num_responses": num_responses,
        "preference": preference,
        "quality_assessment_complete": False,  # Always start with quality assessment - Updated field name
        "quality_assessment_results": None,    # Updated field name
        "context": context,
        "response1": response1,
        "response2": response2,
    }
    
    return DatumSpec(
        message_log=message_log,
        length=total_length,
        extra_env_info=metadata,
        loss_multiplier=loss_multiplier,
        idx=idx,
        task_name="two_stage_genrm",
    )


# ========================= TRAINING INTEGRATION =========================
def setup_two_stage_training(config, tokenizer, dataset, val_dataset):
    """Setup two-stage unified quality assessment training pipeline."""
    
    # Setup base components
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
    ) = setup(config, tokenizer, dataset, val_dataset)

    return (
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
    )

def setup_two_stage_data(tokenizer, data_config, env_configs):
    """Setup data for two-stage unified quality assessment training."""
    
    print("\n▶ Setting up two-stage data...")
    
    # Create task spec for two-stage GenRM
    two_stage_task_spec = TaskDataSpec(
        task_name="two_stage_genrm",
        prompt_file=data_config.get("prompt_file"),
        system_prompt_file=data_config.get("system_prompt_file"),
    )
    
    # Load datasets
    train_data_path = data_config.get("train_data_path")
    val_data_path = data_config.get("val_data_path")
    train_dataset = LocalDataset(train_data_path, task_name="two_stage_genrm", shuffle_seed=data_config.get("shuffle_seed_for_training"), split='train')
    val_dataset = LocalDataset(val_data_path, task_name="two_stage_genrm", split='validation') if val_data_path else None
    
    # Setup task data processors
    task_data_processors = defaultdict(lambda: (two_stage_task_spec, two_stage_genrm_data_processor))
    task_data_processors["two_stage_genrm"] = (two_stage_task_spec, two_stage_genrm_data_processor)
    
    # Setup two-stage environment
    from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env

    two_stage_env = BRRMEnvironment.options(
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.brrm_environment.BRRMEnvironment"
            ),
            "env_vars": dict(os.environ),
        }
    ).remote(env_configs.get("two_stage_genrm", {}))
    
    # Update task to environment mapping
    task_to_env = defaultdict(lambda: two_stage_env)
    task_to_env["two_stage_genrm"] = two_stage_env
    
    val_task_to_env = defaultdict(lambda: two_stage_env)
    val_task_to_env["two_stage_genrm"] = two_stage_env

    # Create processed datasets
    processed_train_dataset = AllTaskProcessedDataset(
        train_dataset,
        tokenizer,
        two_stage_task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )
    
    processed_val_dataset = None
    if val_dataset:
        processed_val_dataset = AllTaskProcessedDataset(
            val_dataset,
            tokenizer,
            two_stage_task_spec,
            task_data_processors,
            max_seq_length=data_config["max_input_seq_length"],
        )
    return processed_train_dataset, processed_val_dataset, task_to_env, val_task_to_env

# ========================= MAIN TRAINING SCRIPT =========================

def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training for Unified Quality Assessment GenRM")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def main():
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_brrm.yaml"
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
    assert config["policy"]["generation"] is not None, "A generation config is required for GRPO"
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    # Setup data
    train_dataset, val_dataset, task_to_env, val_task_to_env = setup_two_stage_data(
        tokenizer, 
        config["data"], 
        config["env"],
    )
    
    # Debug: Print first example from the dataset
    if len(train_dataset) > 0:
        print("\n[DEBUG] First example from dataset:")
        first_example = train_dataset[0]
        print(f"  Keys: {list(first_example.keys())}")
        for key, value in first_example.items():
            if isinstance(value, str) and len(value) > 200:
                print(f"  {key}: {value[:200]}...")
            else:
                print(f"  {key}: {value}")

    # Setup two-stage training
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
    ) = setup_two_stage_training(config, tokenizer, train_dataset, val_dataset)
    

    # Run GRPO training with task environments
    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,  # Pass task_to_env instead of a single env
        val_task_to_env,  # Pass val_task_to_env
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )

if __name__ == "__main__":
    main()