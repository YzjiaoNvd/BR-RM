# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from typing import Any, Optional

from datasets import Dataset, load_dataset, concatenate_datasets
from nemo_rl.data.interfaces import TaskDataSpec
import json
import numpy as np
import torch
import random
import glob
import os 

random.seed(42)



def format_rmbench_example(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Format RM-Bench data for GenRM evaluation while preserving RM-Bench structure."""
    
    prompt_text = data.get("prompt", "")
    chosen_responses = data.get("chosen", [])
    rejected_responses = data.get("rejected", [])
    domain = data.get("domain", "unknown")
    sample_id = data.get("id", "")
    
    # Ensure we have exactly 3 chosen and 3 rejected responses
    assert len(chosen_responses) == 3 and len(rejected_responses) == 3

    # Format as conversation context
    context = f"User: {prompt_text}"
    
    # Create all 9 comparisons (3 chosen x 3 rejected) for the 3x3 matrix
    examples = []
    for i, chosen_resp in enumerate(chosen_responses):
        for j, rejected_resp in enumerate(rejected_responses):
            # Randomly shuffle response order to avoid position bias
            preference = random.choice([0, 1])
            
            if preference == 0: # First response (chosen) is better
                response1 = chosen_resp
                response2 = rejected_resp
            else: # Second response (chosen) is better
                response1 = rejected_resp
                response2 = chosen_resp
            
            example = {
                "num_responses": 2,
                "preference": preference+1,
                "context": context,
                "response1": response1,
                "response2": response2,
                "domain": domain,
                "sample_id": sample_id,
                "chosen_style_idx": i,  # 0=concise, 1=detailed_plain, 2=detailed_markdown
                "rejected_style_idx": j,
            }
            examples.append(example)

    return examples





def format_rewardbench_example(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Format RM-Bench data for GenRM evaluation."""
    # Extract prompt and responses
    prompt_text = data.get("prompt", "")
    chosen_resp = data.get("chosen", "")
    rejected_resp = data.get("rejected", "")
    subset = data.get("subset", "")
    sample_id = data.get("id", "")

    # Format as conversation context
    context = f"User: {prompt_text}"
    
    # Create comparisons between each chosen and rejected response

    preference = random.choice([0, 1])
            
    if preference == 0: # First response (chosen) is better
        response1 = chosen_resp
        response2 = rejected_resp
    else: # Second response (chosen) is better
        response1 = rejected_resp
        response2 = chosen_resp

    domain_map = {
        "alpacaeval-easy": "Chat", 
        "alpacaeval-length": "Chat", 
        "alpacaeval-hard": "Chat", 
        "mt-bench-easy": "Chat", 
        "mt-bench-med": "Chat",
        "mt-bench-hard": "Chat_Hard", 
        "llmbar-natural": "Chat_Hard", 
        "llmbar-adver-neighbor": "Chat_Hard", 
        "llmbar-adver-GPTInst": "Chat_Hard", 
        "llmbar-adver-GPTOut": "Chat_Hard", 
        "llmbar-adver-manual": "Chat_Hard",
        "refusals-dangerous": "Safety", 
        "refusals-offensive": "Safety", 
        "xstest-should-refuse": "Safety", 
        "xstest-should-respond": "Safety", 
        "donotanswer": "Safety",
        "math-prm": "Reasoning", 
        "hep-cpp": "Reasoning", 
        "hep-go": "Reasoning", 
        "hep-java": "Reasoning", 
        "hep-js": "Reasoning", 
        "hep-python": "Reasoning", 
        "hep-rust": "Reasoning"
    }  

    example = {
        "num_responses": 2,
        "preference": preference+1,
        "context": context,
        "response1": response1,
        "response2": response2,
        "domain": domain_map[subset],
        "sample_id": sample_id
    }

    return [example]



def format_rmb_example(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Format RMB data for GenRM evaluation."""
    # Extract conversation context
    conversation_input = data.get("conversation_input")
    context_parts = []
    for turn in conversation_input:
        role = turn.get("role")
        content = turn.get("content")
        if role.lower() == "user":
            context_parts.append(f"User: {content}")
        elif role.lower() == "assistant":
            context_parts.append(f"Assistant: {content}")
    
    context = "\n".join(context_parts)
    
    category_path = data.get("category_path")
    domain = "/".join(category_path.split("/")[:2])
    
    if "Pairwise_set" in domain:
        chosen_resps = [data.get("chosen").get("answer")]
        rejected_resps = [data.get("reject").get("answer")]
        sample_id = data.get("pair_uid")
    else: #BoN_set in domain
        chosen_resps = [data.get("bon_best").get("answer")]
        rejected_list = data.get("loser_list")
        rejected_resps = [each.get("answer") for each in rejected_list]
        sample_id = data.get("bon_uid")
    
    examples = []
    for chosen_resp in chosen_resps:
        for rejected_resp in rejected_resps:
            # Create comparisons with random order to avoid position bias
            preference = np.random.choice([0, 1])
            
            if preference == 0:  # First response (chosen) is better
                response1 = chosen_resp
                response2 = rejected_resp
            else:  # Second response (chosen) is better
                response1 = rejected_resp
                response2 = chosen_resp
            
            example = {
                "num_responses": 2,
                "preference": preference+1,
                "context": context,
                "response1": response1,
                "response2": response2,
                "category_path": category_path,
                "domain": domain,
                "sample_id": sample_id,
            }
            examples.append(example)
    
    return examples


class RMBDataset:
    """RMB dataset for GenRM evaluation."""
    
    def __init__(self, data_folder: str = "BR-RM/dataset/RMB-Reward-Model-Benchmark/RMB_dataset"):
        
        # Find all JSON files in the data folder and subdirectories
        json_files = glob.glob(os.path.join(data_folder, "**", "*.json"), recursive=True)
        
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {data_folder}")
        
        print(f"Found {len(json_files)} JSON files: {[os.path.basename(f) for f in json_files]}")
        
        # Load and combine all data
        all_data = []
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        all_data.extend(file_data)
                    else:
                        all_data.append(file_data)
                print(f"Loaded {len(file_data) if isinstance(file_data, list) else 1} examples from {os.path.basename(json_file)}")
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
        
        print(f"Total loaded examples: {len(all_data)}")
        
        # Format all examples
        all_formatted_examples = []
        for example in all_data:
            formatted_examples = format_rmb_example(example)
            all_formatted_examples.extend(formatted_examples)
        
        print(f"Total formatted examples: {len(all_formatted_examples)}")
        
        # Create a new dataset from the formatted examples
        self.formatted_ds = Dataset.from_list(all_formatted_examples)




class RMBenchDataset:
    """RM-Bench dataset for GenRM evaluation."""
    
    def __init__(self):
        # Load both splits
        ds = load_dataset("THU-KEG/RM-Bench", split="train")
        # ds = ds.select(range(20))
        # Manually expand the dataset by iterating through each example
        all_formatted_examples = []
        for example in ds:
            # Get the three formatted examples for this sample
            formatted_examples = format_rmbench_example(example)
            all_formatted_examples.extend(formatted_examples)
        
        # Create a new dataset from the expanded examples
        self.formatted_ds = Dataset.from_list(all_formatted_examples)







class RewardBenchDataset:
    """RewardBench dataset for GenRM evaluation."""
    
    def __init__(self):
        # Load all splits except Ties
        ds = load_dataset("allenai/reward-bench", split="filtered")

        # Manually expand the dataset by iterating through each example
        all_formatted_examples = []
        for example in ds:
            # Get the three formatted examples for this sample
            formatted_examples = format_rewardbench_example(example)
            all_formatted_examples.extend(formatted_examples)
        
        # Create a new dataset from the expanded examples
        self.formatted_ds = Dataset.from_list(all_formatted_examples)




class LocalDataset(torch.utils.data.Dataset):
    """Dataset for loading local data from local JSONL files."""
    def __init__(self, data_path, task_name: str="genrm", shuffle_seed: int = -1, split: str="validation", max_number: int=-1):
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                one = json.loads(line)
                if one["args"]["num_responses"] == 2:
                    examples = []
                    preference = one["args"]["preference"]

                    example = {
                        "num_responses": 2,
                        "preference": preference, 
                        "context": one["args"]["context"],
                        "response1": one["args"]["response1"],
                        "response2": one["args"]["response2"],
                    }
                    examples.append(example)

                    example = {
                        "num_responses": 2,
                        "preference": 3 - preference, 
                        "context": one["args"]["context"],
                        "response1": one["args"]["response2"],
                        "response2": one["args"]["response1"],
                    }
                    examples.append(example)

                    if split == "validation":
                        data.append(random.choice(examples))
                    else:
                        data += examples
        
        if shuffle_seed != -1:
            rng = np.random.default_rng(shuffle_seed)
            rng.shuffle(data)
            print(f"Shuffled the dataset with {len(data)} samples using seed {shuffle_seed}")

        if max_number != -1:
            data = data[:max_number]
            
        self.data = Dataset.from_list(data)
        self.formatted_ds = self.data
        self.task_name = task_name

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx].copy()
        item["task_name"] = self.task_name
        return item