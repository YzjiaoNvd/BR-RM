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


# Add this new class for hallucination detection datasets
class HallucinationDataset(torch.utils.data.Dataset):
    """Dataset for hallucination detection training."""
    
    def __init__(self, data_path: str, task_name: str = "hallucination", 
                 shuffle_seed: int = -1, split: str = "train", max_number: int = -1):
        data = []
        
        # Load data from JSON file (format from synthetic data generation)
        with open(data_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Process each item in the JSON array
        for item in json_data:
            conversation = item.get("conversation")
                
            context_messages = []
            for msg in conversation:
                context_messages.append(f"{msg['role']}: {msg['content']}")
            context = "\n".join(context_messages)

            response = item.get("response")
            if isinstance(response, list):
                response = "\n".join(response)
    
            ground_truth_errors = item.get("factual_errors")
            error_list = [e['key_element'] for e in ground_truth_errors]
            error_str = " ### ".join(error_list)

            example = {
                "context": context,
                "response": response,
                "ground_truth_errors": error_str,
                "task_name": task_name
            }

            data.append(example)
                    

        
        if shuffle_seed != -1:
            rng = np.random.default_rng(shuffle_seed)
            rng.shuffle(data)
            print(f"Shuffled the dataset with {len(data)} samples using seed {shuffle_seed}")
        
        if max_number != -1:
            data = data[:max_number]
        
        self.data = data
        self.formatted_ds = Dataset.from_list(data)
        self.task_name = task_name
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx].copy()
        item["task_name"] = self.task_name
        return item
