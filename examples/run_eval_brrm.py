# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import argparse
import json
import os
import pprint
from typing import Any, Optional, TypedDict

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data.datasets import AllTaskProcessedDataset, eval_collate_fn
from nemo_rl.data.hf_datasets.reward_benchmarks import (
    RMBenchDataset,
    RewardBenchDataset,
    RMBDataset,
)
from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides

# Import two-stage environment components
from nemo_rl.environments.brrm_environment import (
    format_unified_analysis_prompt,  # Updated function name
    format_scoring_stage_prompt,
    parse_scoring_response,
    parse_unified_analysis_response,  # Updated function name
)


class GenRMEvalConfig(TypedDict):
    dataset_name: str
    batch_size: int
    seed: int
    output_file: str
    use_two_stage: bool


class MasterConfig(TypedDict):
    generation: dict
    data: dict
    eval: GenRMEvalConfig
    tokenizer: dict
    cluster: dict


def genrm_eval_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process evaluation data for unified quality assessment GenRM format."""
    # Extract data
    context = datum_dict.get("context", "")
    response1 = datum_dict.get("response1", "")
    response2 = datum_dict.get("response2", "")
    
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
    token_length = len(user_message["token_ids"])

    message_log.append(user_message)

    # Extract metadata for two-stage evaluation
    metadata = datum_dict.copy()

    # Debug: Print extracted metadata
    if idx < 3:
        print(f"  Extracted metadata: {metadata}")

    return DatumSpec(
        message_log=message_log,
        length=token_length,
        extra_env_info=metadata,
        loss_multiplier=1.0,
        idx=idx,
        task_name="genrm_eval",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run Unified Quality Assessment GenRM two-stage evaluation")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--dataset", type=str, 
                       choices=["rmbench", "localdata", "rewardbench", "rmb"],
                       default=None, help="Dataset to evaluate on")
    args, overrides = parser.parse_known_args()
    return args, overrides


def setup_data(tokenizer, data_config, dataset_name):
    """Set up evaluation dataset."""
    print(f"Setting up {dataset_name} dataset...")
    
    # Load dataset based on type
    dataset_loaders = {
        "rmbench": RMBenchDataset,
        "rewardbench": RewardBenchDataset,
        "rewardbench2": RewardBench2Dataset,
        "localdata": LocalDataset,
        "rmb": RMBDataset,
    }
    
    if dataset_name not in dataset_loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataset_loader = dataset_loaders[dataset_name]()
    test_dataset = dataset_loader.formatted_ds
    
    if test_dataset is None or len(test_dataset) == 0:
        raise ValueError(f"{dataset_name} dataset is empty or failed to load")
    
    print(f"Loaded {len(test_dataset)} examples")
    
    # Create task spec and processed dataset
    eval_task_spec = TaskDataSpec(task_name="genrm_eval")
    processed_dataset = AllTaskProcessedDataset(
        test_dataset,
        tokenizer,
        eval_task_spec,
        genrm_eval_data_processor,
        max_seq_length=data_config["max_input_seq_length"],
    )
    
    return processed_dataset, dataset_loader


def run_two_stage_evaluation(vllm_generation, dataloader, tokenizer, output_file):
    """Run two-stage evaluation: quality assessment then scoring."""
    results = []
    
    print("Running two-stage evaluation...")
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        try:
            # STAGE 1: Quality assessment
            quality_prompts = []
            for metadata in batch["extra_env_info"]:
                context = metadata.get("context", "")
                response1 = metadata.get("response1", "")
                response2 = metadata.get("response2", "")
                
                quality_prompt = format_unified_analysis_prompt(context, response1, response2)  # Updated function call
                user_message = {
                    "role": "user",
                    "content": quality_prompt,
                }
                quality_message = tokenizer.apply_chat_template(  # type: ignore
                    [user_message],
                    tokenize=False,
                    add_generation_prompt=True,
                    add_special_tokens=False,
                )
                quality_prompts.append(quality_message)
            
            # Collect the quality assessment results
            stage1_inputs = BatchedDataDict({"prompts": quality_prompts})
            stage1_outputs = vllm_generation.generate_text(stage1_inputs)
            quality_responses = stage1_outputs.get("texts", [""] * len(quality_prompts))

            updated_quality_responses = []
            for quality_response in quality_responses:
                updated_quality_responses.append(quality_response + "\n" + tokenizer.eos_token + "\n")
            

            # STAGE 2: Scoring
            scoring_prompts = []
            for i, metadata in enumerate(batch["extra_env_info"]):
                context = metadata.get("context", "")
                response1 = metadata.get("response1", "")
                response2 = metadata.get("response2", "")
                
                quality_result = quality_responses[i]
                scoring_prompt = format_scoring_stage_prompt(
                    context, response1, response2, quality_result
                )
                user_message = {
                    "role": "user",
                    "content": scoring_prompt,
                }
                scoring_message = tokenizer.apply_chat_template(  # type: ignore
                    [user_message],
                    tokenize=False,
                    add_generation_prompt=True,
                    add_special_tokens=False,
                )
                scoring_prompts.append(scoring_message)
            
            # Generate scoring responses
            two_stage_prompts = []
            for quality_prompt, updated_quality_response, scoring_prompt in zip(quality_prompts, updated_quality_responses, scoring_prompts):
                two_stage_prompts.append(quality_prompt + updated_quality_response + scoring_prompt)
            stage2_inputs = BatchedDataDict({"prompts": two_stage_prompts})
            stage2_outputs = vllm_generation.generate_text(stage2_inputs)
            scoring_responses = stage2_outputs.get("texts", [""] * len(scoring_prompts))
            
            # Process results
            for idx, (quality_resp, scoring_resp, metadata) in enumerate(zip(
                quality_responses, scoring_responses, batch["extra_env_info"]
            )):
                result = {
                    "idx": batch["idx"][idx].item() if torch.is_tensor(batch["idx"][idx]) else batch["idx"][idx],
                    "quality_response": quality_resp,     # Updated field name
                    "scoring_response": scoring_resp,
                    "metadata": metadata,
                }
                
                # Parse scoring response for metrics
                is_valid, ranking, error_msg = parse_scoring_response(
                    scoring_resp, metadata.get("num_responses", 2)
                )
                
                result["scoring_parse_success"] = is_valid
                if is_valid and ranking:
                        result["predicted_ranking"] = int(ranking)
                else:
                    result["scoring_parse_error"] = error_msg
                
                results.append(result)
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            # Create fallback results
            for idx in range(len(batch["message_log"])):
                results.append({
                    "idx": batch["idx"][idx].item() if torch.is_tensor(batch["idx"][idx]) else batch["idx"][idx],
                    "error": str(e),
                    "scoring_parse_success": False,
                })
    
    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    calculate_metrics(results)


def calculate_metrics(results):
    """Calculate evaluation metrics."""
    total_samples = len(results)
    successful_parses = sum(1 for r in results if r.get("scoring_parse_success", False))
    
    # Calculate ranking accuracy
    correct_rankings = 0
    total_rankings = 0
    
    for result in results:
        if result.get("scoring_parse_success", False) and "predicted_ranking" in result:
            total_rankings += 1
            true_pref = result["metadata"].get("preference")
            if true_pref is not None:
                pred_pref = result["predicted_ranking"]
                if pred_pref == true_pref:
                    correct_rankings += 1
    
    print(f"\nEvaluation Metrics:")
    print(f"  Total samples: {total_samples}")
    print(f"  Successful parses: {successful_parses} ({successful_parses/total_samples:.2%})")
    if total_rankings > 0:
        print(f"  Ranking accuracy: {correct_rankings/total_rankings:.2%} ({correct_rankings}/{total_rankings})")
    else:
        print(f"  No valid rankings found")


def main():
    args, overrides = parse_args()
    
    # Load configuration
    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "evals/brrm_eval.yaml")
    
    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    
    # Apply overrides
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    
    # Override dataset if specified
    if args.dataset:
        config["eval"]["dataset_name"] = args.dataset
    
    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    
    print("Final config:")
    pprint.pprint(config)
    
    # Initialize
    set_seed(config["eval"]["seed"])
    init_ray()
    
    # Setup tokenizer and generation config
    tokenizer = get_tokenizer(config["tokenizer"])
    config["generation"] = configure_generation_config(
        config["generation"], tokenizer, is_eval=True
    )
    
    # Setup data
    dataset, _ = setup_data(tokenizer, config["data"], config["eval"]["dataset_name"])
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        collate_fn=eval_collate_fn,
    )
    
    # Setup cluster and vLLM
    print("Setting up compute cluster...")
    cluster = RayVirtualCluster(
        name="genrm_eval_cluster",
        bundle_ct_per_node_list=[config["cluster"]["gpus_per_node"]] * config["cluster"]["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=config["cluster"]["gpus_per_node"],
        max_colocated_worker_groups=1,
    )
    
    print("Setting up vLLM generation...")
    vllm_generation = VllmGeneration(cluster=cluster, config=config["generation"])
    vllm_generation.prepare_for_generation()
    
    try:
        # Run evaluation
        run_two_stage_evaluation(vllm_generation, dataloader, tokenizer, config["eval"]["output_file"])
    finally:
        # Cleanup
        vllm_generation.finish_generation()
        vllm_generation.shutdown()


if __name__ == "__main__":
    main()