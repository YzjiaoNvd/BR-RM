# Hallucination Detection Evaluation Script
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
from nemo_rl.data.hf_datasets.fact_benchmark import HallucinationDataset
from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides

from nemo_rl.environments.fact_environment import (
    format_claim_extraction_prompt,
    format_error_detection_prompt,
    parse_claim_extraction_response,
    parse_error_detection_response,
    calculate_f1_score,
    GoogleSearchClient,
)

class HallucinationEvalConfig(TypedDict):
    dataset_name: str
    batch_size: int
    seed: int
    output_file: str
    use_google_search: bool

def run_hallucination_evaluation(vllm_generation, dataloader, tokenizer, google_client, output_file):
    """Run two-stage hallucination detection evaluation."""
    results = []
    
    print("Running hallucination detection evaluation...")
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        try:
            # STAGE 1: Claim extraction
            claim_prompts = []
            for metadata in batch["extra_env_info"]:
                context = metadata.get("context", "")
                response = metadata.get("response", "")
                
                claim_prompt = format_claim_extraction_prompt(context, response)
                user_message = {"role": "user", "content": claim_prompt}
                claim_message = tokenizer.apply_chat_template(
                    [user_message],
                    tokenize=False,
                    add_generation_prompt=True,
                    add_special_tokens=False,
                )
                claim_prompts.append(claim_message)
            
            # Generate claim extractions
            stage1_inputs = BatchedDataDict({"prompts": claim_prompts})
            stage1_outputs = vllm_generation.generate_text(stage1_inputs)
            claim_responses = stage1_outputs.get("texts", [""] * len(claim_prompts))
            
            # Process claims and perform searches
            all_search_results = []
            for claim_response in claim_responses:
                is_parsed, extracted_claims, _ = parse_claim_extraction_response(claim_response)
                
                if is_parsed and google_client:
                    # Perform searches
                    search_results = []
                    for claim in extracted_claims[:5]:  # Limit to 5 claims
                        search_terms = claim.get("search_terms", "")
                        if search_terms:
                            try:
                                results = google_client.search(search_terms, num_results=2)
                                formatted = format_google_results(results, search_terms)
                                search_results.append({
                                    "claim": claim["claim"],
                                    "key_element": claim["key_element"],
                                    "search_terms": search_terms,
                                    "search_results": formatted
                                })
                            except Exception as e:
                                print(f"Search error: {e}")
                else:
                    search_results = []
                
                all_search_results.append(search_results)
            
            # STAGE 2: Error detection
            error_prompts = []
            for search_results in all_search_results:
                if search_results:
                    error_prompt = format_error_detection_prompt(search_results)
                else:
                    error_prompt = "No search results available."
                
                user_message = {"role": "user", "content": error_prompt}
                error_message = tokenizer.apply_chat_template(
                    [user_message],
                    tokenize=False,
                    add_generation_prompt=True,
                    add_special_tokens=False,
                )
                error_prompts.append(error_message)
            
            # Generate error detections
            stage2_inputs = BatchedDataDict({"prompts": error_prompts})
            stage2_outputs = vllm_generation.generate_text(stage2_inputs)
            error_responses = stage2_outputs.get("texts", [""] * len(error_prompts))
            
            # Process results and calculate F1 scores
            for idx, (claim_resp, error_resp, metadata) in enumerate(zip(
                claim_responses, error_responses, batch["extra_env_info"]
            )):
                is_valid, detected_errors, _ = parse_error_detection_response(error_resp)
                
                result = {
                    "idx": batch["idx"][idx].item() if torch.is_tensor(batch["idx"][idx]) else batch["idx"][idx],
                    "claim_extraction_response": claim_resp,
                    "error_detection_response": error_resp,
                    "detected_errors": detected_errors if is_valid else [],
                    "metadata": metadata,
                }
                
                # Calculate F1 score if ground truth available
                if metadata.get("ground_truth_errors"):
                    f1_score = calculate_f1_score(
                        detected_errors if is_valid else [],
                        metadata["ground_truth_errors"]
                    )
                    result["f1_score"] = f1_score
                
                results.append(result)
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            # Create fallback results
            for idx in range(len(batch["message_log"])):
                results.append({
                    "idx": batch["idx"][idx].item() if torch.is_tensor(batch["idx"][idx]) else batch["idx"][idx],
                    "error": str(e),
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
    samples_with_f1 = [r for r in results if "f1_score" in r]
    
    if samples_with_f1:
        f1_scores = [r["f1_score"] for r in samples_with_f1]
        mean_f1 = sum(f1_scores) / len(f1_scores)
        
        print(f"\nEvaluation Metrics:")
        print(f"  Total samples: {total_samples}")
        print(f"  Samples with F1 scores: {len(samples_with_f1)}")
        print(f"  Mean F1 score: {mean_f1:.3f}")
        print(f"  Max F1 score: {max(f1_scores):.3f}")
        print(f"  Min F1 score: {min(f1_scores):.3f}")
    else:
        print(f"\nNo F1 scores calculated (no ground truth available)")

def format_google_results(results, query):
    """Format Google search results."""
    if not results:
        return f"No results found for: {query}"
    
    formatted = [f"Search results for: {query}\n"]
    for i, result in enumerate(results):
        title = result.get('title', 'No title').strip()[:100]
        snippet = result.get('snippet', 'No snippet').strip()[:500]
        link = result.get('link', 'No link')
        formatted.append(f"Result {i+1}: {title}\n{snippet}\nSource: {link}\n")
    
    return '\n'.join(formatted)

def parse_args():
    parser = argparse.ArgumentParser(description="Run Hallucination Detection evaluation")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--dataset", type=str, default="localdata", help="Dataset to evaluate on")
    args, overrides = parser.parse_known_args()
    return args, overrides

def main():
    args, overrides = parse_args()
    
    # Load configuration
    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "evals/hallucination_eval.yaml")
    
    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    
    config = OmegaConf.to_container(config, resolve=True)
    
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
    
    # Initialize Google Search Client if configured
    google_client = None
    if config["eval"].get("use_google_search"):
        try:
            from API_KEY import GOOGLE_API_KEY, GOOGLE_SEARCH_ENGINE_ID
            google_client = GoogleSearchClient(
                GOOGLE_API_KEY,
                GOOGLE_SEARCH_ENGINE_ID,
                fetch_full_content=False
            )
            print("Google Search Client initialized")
        except Exception as e:
            print(f"Failed to initialize Google Search Client: {e}")
    
    # Setup data
    dataset = HallucinationDataset(
        config["data"]["val_data_path"],
        task_name="hallucination_detection",
        split="validation"
    )
    
    # Create task spec and processed dataset
    eval_task_spec = TaskDataSpec(task_name="hallucination_eval")
    
    # Import the data processor
    from run_grpo_hallucination import hallucination_data_processor
    
    processed_dataset = AllTaskProcessedDataset(
        dataset,
        tokenizer,
        eval_task_spec,
        hallucination_data_processor,
        max_seq_length=config["data"]["max_input_seq_length"],
    )
    
    # Create dataloader
    dataloader = DataLoader(
        processed_dataset,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        collate_fn=eval_collate_fn,
    )
    
    # Setup cluster and vLLM
    print("Setting up compute cluster...")
    cluster = RayVirtualCluster(
        name="hallucination_eval_cluster",
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
        run_hallucination_evaluation(
            vllm_generation, 
            dataloader, 
            tokenizer, 
            google_client,
            config["eval"]["output_file"]
        )
    finally:
        # Cleanup
        vllm_generation.finish_generation()
        vllm_generation.shutdown()

if __name__ == "__main__":
    main()