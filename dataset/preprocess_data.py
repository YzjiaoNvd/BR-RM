import json
from datasets import load_dataset
import random
from tqdm import tqdm
# from constant import *  # Import your constants if needed

random.seed(42)

def generate_context_from_hf(context):
    conversation_history = ""
    for turn in context:
        if turn['role'] == 'user':
            conversation_history += f"User: {turn['content']}\n"
        else:
            conversation_history += f"Assistant: {turn['content']}\n"
    return conversation_history



def process_hs3_dataset(split):

    dataset_name = "nvidia/HelpSteer3"
    # Load the dataset
    print(f"Loading HuggingFace dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)
    print(f"Loaded {len(dataset)} samples")
    
    all_data = list(dataset)
    
    print("Processing data...")
    
    cnt_skipped = 0
    cnt_processed = 0
    
    processed_data = []
    for data in tqdm(all_data, desc="Processing"):
        args = {}
        args["question_id"] = f"hs3_{cnt_processed}"
        if 'overall_preference' in data:
            if data["overall_preference"] == 0:
                cnt_skipped += 1
                continue
            else:
                args["preference"] = 1 if data["overall_preference"] < 0 else 2
        else: 
            cnt_skipped += 1
            continue

        args["response1"] = data["response1"]
        args["response2"] = data["response2"]
        args["num_responses"] = 2
        args["context"] = generate_context_from_hf(data['context'])
        args["domain"] = data["domain"]

        output_record = {
            "args": args,
            "task_name": "genrm",
        }
        processed_data.append(output_record)
        cnt_processed += 1

    print(f"Total processed samples: {cnt_processed}")
    return processed_data
            



def process_skywork_dataset():

    dataset_name = "Skywork/Skywork-Reward-Preference-80K-v0.2"
    split = "train"

    print(f"Loading HuggingFace dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)
    print(f"Loaded {len(dataset)} samples")
    
    all_data = list(dataset)
    
    print("Processing data...")
    processed_data = []
    cnt_processed = 0

    for data in tqdm(all_data, desc="Processing"):
        if data['source'] == "magpie_ultra":
            continue

        args = {}

        args["question_id"] = f"skywork80k_{cnt_processed}"

        overall_preference = random.choice([1, 2])
        args["preference"] = overall_preference
        
        assert len(data["chosen"]) == len(data["rejected"])
        assert len(data["chosen"]) >= 2

        if args["preference"] == 1:
            args["response1"] = data["chosen"][-1]["content"]
            args["response2"] = data["rejected"][-1]["content"]
        else:
            args["response1"] = data["rejected"][-1]["content"]
            args["response2"] = data["chosen"][-1]["content"]
        
        args["num_responses"] = 2
        context = data["chosen"][:-1]
        args["context"] = generate_context_from_hf(context)
        args["source"] = data["source"]

        output_record = {
            "args": args,
            "task_name": "genrm",
        }
        processed_data.append(output_record)
        cnt_processed += 1

    print(f"Total processed samples: {cnt_processed}")
    return processed_data
            



def process_codepref_dataset():

    dataset_name = "Vezora/Code-Preference-Pairs"
    split = "train"

    print(f"Loading HuggingFace dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)
    print(f"Loaded {len(dataset)} samples")

    shuffled = dataset.shuffle(seed=42)
    subset_15k = shuffled.select(range(8000))

    all_data = list(subset_15k)
    
    print("Processing data...")
    
    processed_data = []
    cnt_processed = 0
    for data in tqdm(all_data, desc="Processing"):
        args = {}
        
        args["question_id"] = f"codepref_{cnt_processed}"

        args["preference"] = random.choice([1, 2])

        if args["preference"] == 1:
            args["response1"] = data["accepted"]
            args["response2"] = data["rejected"]
        else:
            args["response1"] = data["rejected"]
            args["response2"] = data["accepted"]

        args["num_responses"] = 2
        context = [{
            "role": "user",
            "content": data["input"]
        }]
        args["context"] = generate_context_from_hf(context)

        output_record = {
            "args": args,
            "task_name": "genrm",
        }
        processed_data.append(output_record)
        cnt_processed += 1

    print(f"Total processed samples: {cnt_processed}")
    return processed_data


def process_math10k_dataset():

    dataset_name = "xinlai/Math-Step-DPO-10K"
    split = "train"

    print(f"Loading HuggingFace dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)
    print(f"Loaded {len(dataset)} samples")

    all_data = list(dataset)
    
    print("Processing data...")
    
    processed_data = []
    cnt_processed = 0
    for data in tqdm(all_data, desc="Processing"):
        args = {}
        
        args["question_id"] = f"math10k_{cnt_processed}"
        
        args["preference"] = random.choice([1, 2])

        if args["preference"] == 1:
            args["response1"] = data['initial_reason_steps'] + data["full_chosen"]
            args["response2"] = data['initial_reason_steps'] + data["full_rejected"]
        else:
            args["response1"] = data['initial_reason_steps'] + data["full_rejected"]
            args["response2"] = data['initial_reason_steps'] + data["full_chosen"]

        args["num_responses"] = 2
        context = [{
            "role": "user",
            "content": data["prompt"]
        }]
        args["context"] = generate_context_from_hf(context)
        
        output_record = {
            "args": args,
            "task_name": "genrm",
        }
        processed_data.append(output_record)
        cnt_processed += 1

    print(f"Total processed samples: {cnt_processed}")
    return processed_data



def process_training_dataset(output_file):
    hs3_data = process_hs3_dataset("train")
    skywork_data = process_skywork_dataset()
    codepref_data = process_codepref_dataset()
    math10k_data = process_math10k_dataset()

    processed_data = hs3_data + skywork_data + codepref_data + math10k_data
    random.shuffle(processed_data)

    with open(output_file, "w") as fout:
        for output_record in processed_data:
            fout.write(json.dumps(output_record) + "\n")
                
    print(f"Processing completed.")
    print(f"Total processed samples: {len(processed_data)}")
    print(f"Output written to: {output_file}")



def process_val_dataset(output_file):
    processed_data = process_hs3_dataset("validation")

    with open(output_file, "w") as fout:
        for output_record in processed_data:
            fout.write(json.dumps(output_record) + "\n")
                
    print(f"Processing completed.")
    print(f"Total processed samples: {len(processed_data)}")
    print(f"Output written to: {output_file}")


if __name__ == "__main__":

    split = "train"
    process_training_dataset(
        output_file=f"./train_data.jsonl"
    )

    split = "val"
    process_val_dataset(
        output_file=f"./val_data.jsonl"
    )
